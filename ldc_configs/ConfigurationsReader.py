from configparser import ConfigParser

class ConfigurationsReader():
    def __init__(self, kind):
        self.config_reader = ConfigParser()
        self.config_reader.read('/media/new_volume/MyProjects/learningPinn/ldc_pinn/ldc_configs/ldc_configurations.ini')
        self.config_data = self.config_reader[kind]

        
    def getLayers(self):
        layers_config = self.config_data['layers']
        layersStringList = layers_config.split(",")
        layersList = list(map(lambda i : int(i), layersStringList))
        return layersList
        
    def getTrainedStateDictPath(self):
        return self.config_data['trained_state_dict_path']
