# The distiller class
# Modified from source: https://www.kaggle.com/code/lonnieqin/knowledge-distillation

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class Distiller:
    def __init__(self, teacher, alpha=1, beta=1):
        """

        @param teacher:
        @param alpha: The weight hyperparameter for the loss object from the student model. Default to 1. It is here
        only to explore the effect when no direct learning from the data.
        @param beta: The weight hyperparameter for the loss object from distillation. To adjust how much the model is
        learning from distillation v.s. direct learning, change this hyperparameter.
        """
        super().__init__()
        self.teacher = teacher
        self.alpha = alpha
        self.beta = beta
        self.results = {}

    # Distillation loss function based on the MSE between the inputs
    # used to measure the loss for regression tasks
    def mse_with_distillation_loss_function(self, x, y):
        teacher_predict = self.teacher.predict(x)

        student_loss = np.sum((y - np.mean(y)) ** 2)
        distillation_loss = np.sum((teacher_predict - np.mean(y)) ** 2)

        loss = self.alpha * student_loss + self.beta * distillation_loss

        return loss


# A wrapper class to adapt the sklearn models to tf models
class ModelWrapper:
    def __init__(self, model):
        super().__init__()
        self.model = model

    def __call__(self, inputs, training=None, mask=None):
        return tf.convert_to_tensor(self.model.predict(inputs.numpy()))
