# -*- coding: utf-8 -*-
# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Evaluation script for mip-NeRF."""
import functools
from os import path

from absl import app
from absl import flags
import flax
from flax.metrics import tensorboard
from flax.training import checkpoints
import jax
from jax import random
import numpy as np

from internal import datasets
from internal import math
from internal import models
from internal import utils
from internal import vis
from run_nerf_helpers import *
import imageio


FLAGS = flags.FLAGS
utils.define_common_flags()
flags.DEFINE_bool(
    'eval_once', True,
    'If True, evaluate the model only once, otherwise keeping evaluating new'
    'checkpoints if any exist.')
flags.DEFINE_bool('save_output', True,
                  'If True, save predicted images to disk.')


def main(unused_argv):
    
  
  #set the lens parameters for interpolation
  a1=0.2
  f1=0.1
  l1=0.67
  pose1=11
    
  a2=0.02
  f2=0.1
  l2=0.67
  pose2=11#0

  train_coc=1
  fix_pose = True

  n_frames = 90
    
    
  config = utils.load_config()
  dataset = datasets.get_dataset('test', FLAGS.data_dir, config)
  model, init_variables = models.construct_mipnerf(
      random.PRNGKey(20200823), dataset.peek())
  optimizer = flax.optim.Adam(config.lr_init).create(init_variables)
  state = utils.TrainState(optimizer=optimizer)
  del optimizer, init_variables


  def render_eval_fn(variables, _, rays, a, f, l, train_coc):
    return jax.lax.all_gather(
        model.apply(
            variables,
            random.PRNGKey(0),  # Unused.
            rays,
            randomized=False,
            white_bkgd=config.white_bkgd, a=a, f=f, l=l, train_coc=train_coc),
        axis_name='batch')

  render_eval_pfn = jax.pmap(
      render_eval_fn,
      in_axes=(None, None, 0, None, None, None, None),  # Only distribute the data input.
      donate_argnums=(2,),
      axis_name='batch',
  )
    

  ssim_fn = jax.jit(functools.partial(math.compute_ssim, max_val=1.))

  last_step = 0
  out_dir = path.join(FLAGS.train_dir,
                      'path_renders' if config.render_path else 'test_preds')
  if not FLAGS.eval_once:
    summary_writer = tensorboard.SummaryWriter(
        path.join(FLAGS.train_dir, 'eval'))
  while True:
    state = checkpoints.restore_checkpoint(FLAGS.train_dir, state)
    step = int(state.optimizer.state.step)
    if step <= last_step:
      continue
    if FLAGS.save_output and (not utils.isdir(out_dir)):
      utils.makedirs(out_dir)
    psnr_values = []
    ssim_values = []
    avg_values = []
    rgbs = []
    rays = []

    if not FLAGS.eval_once:
      showcase_index = random.randint(random.PRNGKey(step), (), 0, dataset.size)
    for idx in range(dataset.size):
        batch = next(dataset)
        rays.append(batch['rays'])
        loss_mult = batch['rays'].lossmult
        ne_ar = batch['rays'].near
        f_ar = batch['rays'].far
    origins = rays[pose2].origins - rays[pose1].origins
    directions = rays[pose2].directions - rays[pose1].directions
    viewdirs = rays[pose2].viewdirs - rays[pose1].viewdirs
    radii = rays[pose2].radii - rays[pose1].radii
    for i in range(n_frames):
        
        theta = i/n_frames

        o = origins * theta + rays[pose1].origins
        d = directions * theta + rays[pose1].directions
        v = viewdirs * theta + rays[pose1].viewdirs
        r = radii * theta + rays[pose1].radii
        
        a= a1*theta + a2*(1-theta) #aperture
        f= f1*theta + f2*(1-theta) #focal length 
        l= l1*theta + l2*(1-theta) #focus distance
        
        
        
        rrays = utils.Rays(
        origins=o,
        directions=d,
        viewdirs=v,
        radii=r,
        lossmult=loss_mult,
        near=ne_ar,
        far=f_ar)
        print(f'New__Evaluating {i+1}/n_frames')
        
        pred_color, pred_distance, pred_acc = models.render_image(
          functools.partial(render_eval_pfn, state.optimizer.target),
          rrays,
          None,
          chunk=FLAGS.chunk, a=a, f=f, l=l, train_coc=1)
        
        rgbs.append(pred_color)
        vis_suite = vis.visualize_suite(pred_distance, pred_acc)
        if jax.host_id() != 0:  # Only record via host 0.
            continue
        if FLAGS.save_output and (config.test_render_interval > 0):
            if (idx % config.test_render_interval) == 0:
              print('Starting Save!!!!!!!!!!!!!!!!!')
              print(out_dir)
              print('color_{:03d}.png'.format(i))
              utils.save_img_uint8(
                  pred_color, path.join(out_dir, 'color_{:03d}.png'.format(i)))
              utils.save_img_float32(
                  pred_distance, path.join(out_dir, 'distance_{:03d}.tiff'.format(i)))
              utils.save_img_float32(
                  pred_acc, path.join(out_dir, 'acc_{:03d}.tiff'.format(i)))
              for k, v in vis_suite.items():
                utils.save_img_uint8(
                    v, path.join(out_dir, k + '_{:03d}.png'.format(i)))

    rgbs = np.stack(rgbs, 0)
    print('Generating video!!!!!!!!!')
    imageio.mimwrite(path.join(out_dir, 'video.mp4'), to8b(rgbs), fps=24, quality=9)
    if FLAGS.eval_once:
      break
    if int(step) >= config.max_steps:
      break
    last_step = step


if __name__ == '__main__':
  app.run(main)
