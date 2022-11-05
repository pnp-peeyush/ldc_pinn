import torch
import torch.nn as nn
import sys
sys.path.append("..")
import ldc_configs

class LdcPinnUtils():
    
    def __init__(self):
        self.device = self.device = torch.device('cuda')
        self.config = ldc_configs.ConfigurationsReader('default')
        self.layers = self.config.getLayers()
    
    def generateNetwork(self):
        retVal = nn.Sequential()
        for i in range(len(self.layers)-1) :
            activation = nn.Tanh()

            layer = nn.Linear(self.layers[i], self.layers[i+1])

            retVal.append(layer)
            if(i != len(self.layers)-2):
                retVal.append(activation)
        
        return retVal.to(device=self.device)
        
    def saveTrainedModel(self, currentLoss, currentEpoch, currentModel, currentOptimizer):
        oldNetwork = self.generateNetwork()
        stateDictPath = self.config.getTrainedStateDictPath()
        currentStateOfTraining = {
            'epoch' : currentEpoch,
            'modelState' : currentModel.state_dict(),
            'bestLoss' : currentLoss,
            'optimizerState' : currentOptimizer.state_dict()
        }
        
        try :
            lastStateOfTraining = torch.load(stateDictPath)
        except FileNotFoundError as fe :
            print("No state dict found at path : {} \n"\
                    "Saving current model instead with loss : {} at epoch {}".format(stateDictPath,currentLoss,currentEpoch))
            torch.save(currentStateOfTraining, stateDictPath)
            return currentStateOfTraining

        if currentLoss < lastStateOfTraining['bestLoss']:
            print("Saving new best model with loss : {} at epoch : {}".format(currentLoss,currentEpoch))
            torch.save(currentStateOfTraining, stateDictPath)
            return currentStateOfTraining
        else:
            print("Not saving current model with loss : {} at epoch : {} "\
                "doesn't beat the best model with loss : {} at epoch : {}".format(currentLoss, currentEpoch,
                lastStateOfTraining['bestLoss'], lastStateOfTraining['epoch'])) 
            return lastStateOfTraining
