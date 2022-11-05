#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
"""
   ldcPinnTrainer.py
   
   Copyright 2022 Unknown <archuser@archuser>
   
   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2 of the License, or
   (at your option) any later version.
   
   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.
   
   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
   MA 02110-1301, USA.
   
   
"""

import ldc_configs as lcon
import ldc_classes as lc

def main(args):    
    configReader = lcon.ConfigurationsReader('default')

    layers = configReader.getLayers()
    boundaryPoints = 2500         #actual number of training points at boundary = 4x this value
    collocationPoints = 350     #actual number of collocation training points = square of this value
    H = 1.
    Uo = 1.
    rho = 1.
    Re = 100
    
    #pinn = lc.LdcPinnTrainer(layers,boundaryPoints,collocationPoints,H,Uo,rho,Re,trainFromScratch = False)

    #pinn.train()
    
    modelValidator = lc.LdcPinnModelValidator(layers)
    modelValidator.generatePlot2()

    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
