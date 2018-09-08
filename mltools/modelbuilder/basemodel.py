class BaseModel():
    """
    Abstract class for building machinelearning models
    """

    def load_datasets(self,*args,**kwargs):
        """
        Loads datasets for model  building
        """
        raise NotImplementedError