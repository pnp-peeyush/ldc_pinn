import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import sys
sys.path.append("..")
import ldc_classes as lc
import ldc_configs as lconf


class LdcPinnModelValidator():
    def __init__(self, layers):
        self.device = torch.device('cuda')
        self.config = lconf.ConfigurationsReader("default")
        self.stateDict = torch.load(self.config.getTrainedStateDictPath())
        self.network = lc.LDC_UTILS.generateNetwork()
        self.network.load_state_dict(self.stateDict['modelState'])

    def generatePlot1(self):
        df = pd.read_csv('/media/new_volume/MyProjects/learningPinn/ldc_pinn/ldc_data/all/plot_u_y_Ghia100.csv')
        x = 0.5*np.ones(100)
        y = np.linspace(0,1,100)
        xy = np.transpose(np.vstack((x,y)))
        xy_tensor = torch.tensor(xy,
                                dtype=torch.float32,
                                requires_grad=False, device=self.device)
        uvp = self.network(xy_tensor)
        u = uvp[:,0].reshape(1,-1)[0].cpu().detach().numpy()
        
        plt.plot(u,y)
        plt.scatter(df['u'],df['y'])
        plt.show()
        
    def generatePlot2(self):
        x = torch.linspace(0,1,500).to(device=self.device)
        y = torch.linspace(0,1,500).to(device=self.device)
        
        # x & t grids:
        X, Y = torch.meshgrid(x, y)

        # x & t columns:
        xcol = X.reshape(-1, 1)
        ycol = Y.reshape(-1, 1)

        # one large column:
        uvpsol = self.network(torch.hstack((xcol, ycol)))

        # reshape solution:
        U = uvpsol[:,0].reshape(x.numel(), y.numel())
        V = uvpsol[:,1].reshape(x.numel(), y.numel())
        P = uvpsol[:,2].reshape(x.numel(), y.numel())

        # transform to numpy:
        xnp = x.cpu().numpy()
        ynp = y.cpu().numpy()
        Unp = np.transpose(U.cpu().detach().numpy())
        Vnp = np.transpose(V.cpu().detach().numpy())
        Pnp = np.transpose(P.cpu().detach().numpy())
        
        # plot:
        fig = plt.figure(figsize=(9, 4.5))
        ax_u = fig.add_subplot(131)
        ax_v = fig.add_subplot(132)
        ax_p = fig.add_subplot(133)

        h = ax_u.imshow(Unp,
                      interpolation='nearest',
                      cmap='rainbow', 
                      extent=[ynp.min(), ynp.max(), xnp.min(), xnp.max()], 
                      origin='lower', aspect='auto')
        ax_v.imshow(Vnp,
                      interpolation='nearest',
                      cmap='rainbow', 
                      extent=[ynp.min(), ynp.max(), xnp.min(), xnp.max()], 
                      origin='lower', aspect='auto')
        ax_p.imshow(Pnp,
                      interpolation='nearest',
                      cmap='rainbow', 
                      extent=[ynp.min(), ynp.max(), xnp.min(), xnp.max()], 
                      origin='lower', aspect='auto')
        divider = make_axes_locatable(ax_u)
        cax = divider.append_axes("right", size="5%", pad=0.10)
        cbar = fig.colorbar(h, cax=cax)
        cbar.ax.tick_params(labelsize=10)
        plt.show()
