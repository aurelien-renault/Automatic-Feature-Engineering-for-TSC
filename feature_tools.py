import numpy as np
from scipy import signal
from featuretools.primitives import TransformPrimitive
from woodwork.column_schema import ColumnSchema

class Derivative(TransformPrimitive):
    
    name = "Derivative"
    input_types = [ColumnSchema(semantic_tags={"numeric"})]
    return_type = ColumnSchema(semantic_tags={"numeric"})
    uses_full_dataframe = True
    
    def __init__(self, order=1):
        self.order = order

    def get_function(self):
        def derivative(values, n=self.order):
            if n == 1:
                return values.diff()
            else:
                return derivative(values.diff(), n = n-1)

        return derivative

class Integral(TransformPrimitive):
    
    name = "Integral"
    input_types = [ColumnSchema(semantic_tags={"numeric"})]
    return_type = ColumnSchema(semantic_tags={"numeric"})
    uses_full_dataframe = True
    
    def __init__(self, order=1):
        self.order = order

    def get_function(self):
        def integral(values, n=self.order):
            if n == 1:
                return values.cumsum()
            else:
                return integral(values.cumsum(), n = n-1)

        return integral

class AutoCorr(TransformPrimitive):
    
    name = "auto_correlation"
    input_types = [ColumnSchema(semantic_tags={"numeric"})]
    return_type = ColumnSchema(semantic_tags={"numeric"})
    uses_full_dataframe = True

    def get_function(self):
        def auto_corr(values):
            return np.correlate(values, values, 'same')

        return auto_corr

class FourrierTransform(TransformPrimitive):

    name = "fourrier_transform"
    input_types = [ColumnSchema(semantic_tags={"numeric"})]
    return_type = ColumnSchema(semantic_tags={"numeric"})
    uses_full_dataframe = True

    def get_function(self):
        def fourrier_tsf(values):
            coeffs = signal.periodogram(values, scaling='spectrum')[1]
            return np.pad(coeffs, (0,len(values)-len(coeffs)), "constant", constant_values=0)
        return fourrier_tsf