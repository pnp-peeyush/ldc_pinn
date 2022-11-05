import numpy as np
import torch
import torch.nn as nn
import sys
sys.path.append("..")
import ldc_classes as lc
import ldc_configs as lconf

class LdcPinnTrainer():
    def __init__(self,layers, wall_points,collocation_points, H, Uo, rho, Re, trainFromScratch = True):
        self.device = torch.device('cuda')
        self.config = lconf.ConfigurationsReader("default")
        self.layers = layers
        self.rho = rho
        self.H = H
        self.Uo = Uo
        self.Re = Re
        self.network = lc.LDC_UTILS.generateNetwork()
        if trainFromScratch != True:
            self.tryOldNetworkState()
        self.x_domain = [0., H]
        self.y_domain = [0., H]
        self.wall_points = wall_points
        self.collocation_points = collocation_points
        self.refreshTrainingData()
        # this optimizer updates the weights and biases of the net:
        self.optimizer = torch.optim.LBFGS(self.network.parameters(),
                                    lr=1,
                                    max_iter=50000,
                                    max_eval=50000,
                                    history_size=50,
                                    tolerance_grad=1e-04,
                                    tolerance_change=0.5 * np.finfo(float).eps,
                                    line_search_fn="strong_wolfe")
                                    
        # typical MSE loss (this is a function):
        self.loss = nn.MSELoss()

        # loss :
        self.ls = 0

        # iteration number:
        self.iter = 0
        
        # null vector to test against f:
        self.null =  torch.zeros((self.x_collocation.shape[0]), device=self.device)

    def tryOldNetworkState(self):
        try :
            oldState = torch.load(self.config.getTrainedStateDictPath())
            self.network.load_state_dict(oldState['modelState'])
            print("Loaded old network state with loss : {} at epoch : {}".format(oldState['bestLoss'], oldState['epoch']))
        except Exception as e:
            print("couldn't load old network state because of below error, continuing training with default weights and biases")
            print(e)
            return 0
    
    def flowVariables(self,x,y): 
        return self.network( torch.hstack((x, y)) )
        
    def costFunction(self, x,y):
        uvp = self.flowVariables(x,y)
        u = uvp[:,0]
        v = uvp[:,1]
        p = uvp[:,2]
        
        dudx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), 
                                    retain_graph=True, create_graph=True)[0].reshape(1,-1)[0]

        dvdx = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), 
                                    retain_graph=True, create_graph=True)[0].reshape(1,-1)[0]

        dpdx = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), 
                                    retain_graph=True, create_graph=False)[0].reshape(1,-1)[0]
        
        dudy = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), 
                                    retain_graph=True, create_graph=True)[0].reshape(1,-1)[0]
        
        dvdy = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), 
                                    retain_graph=True, create_graph=True)[0].reshape(1,-1)[0]

        dpdy = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), 
                                    retain_graph=True, create_graph=False)[0].reshape(1,-1)[0]
        
        d2udx2 = torch.autograd.grad(dudx, x, grad_outputs=torch.ones_like(dudx), 
                                    retain_graph=True, create_graph=False)[0].reshape(1,-1)[0]
        
        d2vdx2 = torch.autograd.grad(dvdx, x, grad_outputs=torch.ones_like(dvdx), 
                                    retain_graph=True, create_graph=False)[0].reshape(1,-1)[0]
        
        d2udy2 = torch.autograd.grad(dudy, y, grad_outputs=torch.ones_like(dudy), 
                                    retain_graph=True, create_graph=False)[0].reshape(1,-1)[0]
       
        d2vdy2 = torch.autograd.grad(dvdy, y, grad_outputs=torch.ones_like(dvdy), 
                                    retain_graph=True, create_graph=False)[0].reshape(1,-1)[0]
        
        l1 = (u*dudx) + (v*dudy) + (dpdx/self.rho) - ((self.Uo*self.H/self.Re)*(d2udx2 + d2udy2))
        l2 = (u*dvdx) + (v*dvdy) + (dpdy/self.rho) - ((self.Uo*self.H/self.Re)*(d2vdx2 + d2vdy2))
        l3 = dudx + dvdy
        
        return torch.vstack((l1,l2,l3))
    
    def getBoundaryConditionTrainingData(self, wall_points, Uo):
        x_low = self.x_domain[0]*np.ones(wall_points)
        #x_mid = np.linspace(self.x_domain[0],self.x_domain[1],wall_points)
        x_mid = (self.x_domain[1]-self.x_domain[0])*np.random.rand(wall_points) + self.x_domain[0]
        x_high = self.x_domain[1]*np.ones(wall_points)
        
        y_low = self.y_domain[0]*np.ones(wall_points)
        #y_mid = np.linspace(self.y_domain[0],self.y_domain[1],wall_points)
        y_mid = (self.y_domain[1]-self.y_domain[0])*np.random.rand(wall_points) + self.y_domain[0]
        y_high = self.y_domain[1]*np.ones(wall_points)
        
        X_west = np.transpose(np.vstack((x_low,y_mid)))
        U_west = np.zeros_like(X_west)
        
        X_east = np.transpose(np.vstack((x_high,y_mid)))
        U_east = np.zeros_like(X_east)
        
        X_south = np.transpose(np.vstack((x_mid,y_low)))
        U_south = np.zeros_like(X_south)
        
        X_north = np.transpose(np.vstack((x_mid,y_high)))
        U_north = np.transpose(np.vstack((Uo*np.ones(wall_points), np.zeros(wall_points))))
        
        X_boundary = np.vstack((X_east,X_west,X_south,X_north))
        U_boundary = np.vstack((U_east,U_west,U_south,U_north))
        
        random_index = np.arange(4*wall_points)
        np.random.shuffle(random_index)
        
        return X_boundary[random_index], U_boundary[random_index]

    def getCollocationTrainingData(self, collocation_points):
        #x_range = np.linspace(self.x_domain[0],self.x_domain[1],collocation_points)
        x_range = (self.x_domain[1]-self.x_domain[0])*np.random.rand(collocation_points) + self.x_domain[0]
        #y_range = np.linspace(self.y_domain[0],self.y_domain[1],collocation_points)
        y_range = (self.y_domain[1]-self.y_domain[0])*np.random.rand(collocation_points) + self.y_domain[0]
        x,y = np.meshgrid(x_range,y_range)
        X_collocation = np.hstack((np.reshape(x,(-1,1)),np.reshape(y,(-1,1))))
        np.random.shuffle(X_collocation)
        return X_collocation
        
    def refreshTrainingData(self):
        X_boundary, U_boundary = self.getBoundaryConditionTrainingData(self.wall_points, self.Uo)
        self.x_boundary = torch.tensor(X_boundary[:,0].reshape(-1, 1),
                                dtype=torch.float32,
                                requires_grad=True, device=self.device)
        self.y_boundary = torch.tensor(X_boundary[:,1].reshape(-1, 1),
                                dtype=torch.float32,
                                requires_grad=True, device=self.device)
        self.u_boundary = torch.tensor(U_boundary[:,0].reshape(-1, 1),
                                dtype=torch.float32,
                                requires_grad=True, device=self.device)
        self.v_boundary = torch.tensor(U_boundary[:,1].reshape(-1, 1),
                                dtype=torch.float32,
                                requires_grad=True, device=self.device)
        
        X_collocation = self.getCollocationTrainingData(self.collocation_points)
        self.x_collocation = torch.tensor(X_collocation[:,0].reshape(-1, 1),
                                dtype=torch.float32,
                                requires_grad=True, device=self.device)
        self.y_collocation = torch.tensor(X_collocation[:,1].reshape(-1, 1),
                                dtype=torch.float32,
                                requires_grad=True, device=self.device)

    def closure(self):
        #print("calling closure ...")
        # reset gradients to zero:
        self.optimizer.zero_grad()

        #self.refreshTrainingData()

        # u & f predictions:
        u_prediction = self.flowVariables(self.x_boundary, self.y_boundary)
        f_prediction = self.costFunction(self.x_collocation, self.y_collocation)

        #print(u_prediction.shape,self.u.shape,f_prediction.shape,self.null.shape)
        # losses:
        u_loss = self.loss(u_prediction[:,0:2], torch.hstack((self.u_boundary,self.v_boundary)))

        f_loss_1 = self.loss(f_prediction[0], self.null)
        f_loss_2 = self.loss(f_prediction[1], self.null)
        f_loss_3 = self.loss(f_prediction[2], self.null)
        self.ls = u_loss + f_loss_1 + f_loss_2 + f_loss_3
        
        # derivative with respect to net's weights:
        self.ls.backward()

        # increase iteration count:
        self.iter += 1

        # print report:
        if not self.iter % 10:
            print('Epoch: {0:}, Loss: {1:6.6f}'.format(self.iter, self.ls))

        return self.ls
           
    def train(self):
        """ training loop """
        self.network.train()

        torch.cuda.empty_cache()

        self.optimizer.step(self.closure)
        
        lc.LDC_UTILS.saveTrainedModel(
                                        self.ls,
                                        self.iter,
                                        self.network,
                                        self.optimizer
                                    )
        #torch.save(self.optimizer.state_dict(),"ldcOptimizerState.pth")
        #torch.save(self.network.state_dict(),"ldcNetworkState.pth")
