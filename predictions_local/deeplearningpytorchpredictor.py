import os
import math
import torch
from prediction_deep_learning.pytorch.deep_learning_pytorch import FlexibleCNNClassifier

class DeeplearningPytorchPredictor:
    """
    Wrapper class for the FlexibleCNNClassifier PyTorch model.
    Loads the model from disk and allows direct prediction of a single label.
    """

    # Map numeric class indices to human-readable labels
    class_map = {
        0: "backward",
        1: "forward",
        2: "landing",
        3: "left",
        4: "right",
        5: "takeoff"
    }

    def __init__(self, model_path=None, num_classes=6, device=None):
        # Use GPU if available
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")

        # Default model path if none provided
        if model_path is None:
            BASE_DIR = os.getcwd()  # Use current working directory
            model_path = os.path.join(
                BASE_DIR,
                "prediction_deep_learning",
                "pytorch",
                "FlexibleCNNClassifier.pth"
            )

        # Initialize and load model
        self.model = FlexibleCNNClassifier(num_classes=num_classes)
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()  # set to evaluation mode

    def __call__(self, X):
        """
        Forward pass on input tensor X and return the first predicted label.
        """
        if not isinstance(X, torch.Tensor):
            raise TypeError("X must be a torch.Tensor")

        X = X.to(self.device)
        with torch.inference_mode():
            logits = self.model(X)
            y_pred_probs = torch.softmax(logits, dim=1)
            predicted_classes = torch.argmax(y_pred_probs, dim=1)

        
        #predicted_classes_txt = [self.class_map[int(p)] for p in predicted_classes]
        #print("All predicted labels:", predicted_classes_txt)

        # Convert to take an average
        predicted_ints = [int(p) for p in predicted_classes]
        print("All predicted classes:", predicted_ints)
        predicted_ints = predicted_ints[0:50]
        print("Predicted classes for average:", predicted_ints)
       
        # Average - round up
        avg_class = math.ceil(sum(predicted_ints) / len(predicted_ints))
        print("Average class:", avg_class)

        # Convert to text label
        pred_label = self.class_map[avg_class]
        print("Average class label:", pred_label)

        # Return the first label
        return pred_label
    
    if __name__ == "__main__":
        DeeplearningPyTorchPredictor()