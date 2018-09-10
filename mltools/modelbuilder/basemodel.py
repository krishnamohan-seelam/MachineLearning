class BaseLoader():
    """
    Abstract class for loading machinelearning datasets
    """

    def load_datasets(self,*args,**kwargs):
        """
        Loads datasets for model  building
        """
        raise NotImplementedError
    
class BasePreprocessor():
    """
    Abstract class for preprocessing machinelearning datasets
    """
    def null_counter(self,*args,**kwargs):
         
         raise NotImplementedError
     