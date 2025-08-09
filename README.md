# Unified Feature Framework for Person Re-Identification

This project implements an end-to-end deep learning system for **Person Re-Identification (Re-ID)** using PyTorch and Google Colab. The core of the project is a `UnifiedNet` model, built upon a ResNet-50 backbone, which is trained to generate highly discriminative feature embeddings. These embeddings allow the system to accurately identify the same person across different, non-overlapping camera views, a critical task in smart surveillance and video analysis.

The entire workflow, from data preparation and model training to evaluation and visual demonstration, is encapsulated within a Google Colab notebook, leveraging free GPU resources for accelerated training.

## Key Features

  * **Modern Deep Learning Backbone:** Utilizes a pre-trained **ResNet-50** to extract powerful base features from images.
  * **Advanced Training Strategy:** Employs **Triplet Margin Loss** to train the network, an effective technique that directly optimizes the embedding space by pushing features from different people apart while pulling features of the same person together.
  * **End-to-End Pipeline:** Provides a complete, runnable pipeline in a Google Colab notebook, covering data setup, training, evaluation, and inference.
  * **Quantitative Evaluation:** Measures model performance using the standard **Rank-1 Accuracy** metric for person Re-ID.
  * **Visual Demonstration:** Includes a practical demo script to visualize the model's search and retrieval capabilities in action.

-----

## Project Structure

The repository is organized to maintain a clean and modular codebase:

```
├── notebooks/
│   └── Unified_MOT_ReID.ipynb      # Main Google Colab notebook
│
├── src/
│   ├── dataloader.py               # Custom PyTorch Dataset for triplets
│   └── model.py                    # UnifiedNet model architecture
│
├── weights/                        # Saved model checkpoints (ignored by .gitignore)
│
├── data/                           # DukeMTMC-reID dataset (ignored by .gitignore)
│
└── README.md                       # Project explanation (this file)
```

-----

## Getting Started

Follow these steps to set up and run the project on your own.

### Prerequisites

  * A Google Account (for Google Colab and Google Drive).
  * A Kaggle Account (to download the dataset via API).

### Step-by-Step Instructions

1.  **Clone the Repository (Optional):**
    You can clone this repository to have a local copy, but all steps can be run directly from the main notebook.

    ```bash
    git clone https://github.com/oophswet/Unified_MOT_ReID.git
    ```

2.  **Open in Google Colab:**
    Open the `Unified_MOT_ReID.ipynb` notebook located in the `/notebooks` directory in Google Colab.

3.  **Configure the Runtime:**
    In the Colab menu, navigate to **Runtime → Change runtime type** and select **GPU** as the hardware accelerator.

4.  **Run the Notebook Cells:**
    Execute the cells in the notebook in order. The notebook is self-contained and will guide you through:

      * **Mounting Google Drive:** To save your dataset and model weights permanently.
      * **Kaggle API Setup:** To download the dataset directly.
      * **Data Download & Preparation:** The notebook will download and extract the **DukeMTMC-reID** dataset.
      * **Model Training:** The training loop will run, saving checkpoints to your Google Drive after each epoch.
      * **Evaluation:** After training, the notebook will calculate the model's Rank-1 Accuracy.
      * **Visual Demo:** The final cells will provide a visual demonstration of the model's performance.

-----

## Training & Evaluation

The model is trained using the **Adam optimizer** and **Triplet Margin Loss**. The training progress, including the loss for each epoch, is printed in the notebook.

The primary evaluation metric used is **Rank-1 Accuracy**, which measures the percentage of times the model correctly identifies an image of a person as the top match from a large gallery of test images.

### Results

After an initial training run of 10 epochs, the model achieved the following baseline performance:

  * **Rank-1 Accuracy:** **11.89%**

*(This serves as a functional baseline. For higher accuracy, see the Future Improvements section.)*

-----

## Future Improvements

The current model provides a solid foundation. The following steps can be taken to significantly enhance its performance:

  * **Train for More Epochs:** The single most effective way to improve accuracy. Training for 60-80 epochs or more will allow the model to learn much more robust features.
  * **Implement a Learning Rate Scheduler:** Gradually reducing the learning rate during training (e.g., using `torch.optim.lr_scheduler.StepLR`) can help the model converge to a better solution.
  * **Experiment with Hyperparameters:** Adjusting the `learning_rate`, `batch_size`, or `embedding_dim` could yield better results.
  * **Use Advanced Triplet Mining:** Instead of random sampling, implement "hard triplet mining" to focus the training on the most difficult examples, which often leads to faster and more effective learning.
