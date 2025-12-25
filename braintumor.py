import sys
import os
import cv2
import io
import json
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader 
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QTabWidget, QFileDialog, QGroupBox, QFormLayout, QMessageBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical



class QTextEditLogger(tf.keras.callbacks.Callback):
    def __init__(self, log_func):
        super().__init__()
        self.log_func = log_func
        self.current_epoch = 0  # Track current epoch
        
    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch + 1    

    def on_epoch_end(self, epoch, logs=None):
        msg = f"Epoch {epoch+1}: loss={logs.get('loss'):.4f}, val_loss={logs.get('val_loss'):.4f}"
        self.log_func(msg)
        QApplication.processEvents()

    def on_train_batch_end(self, batch, logs=None):
        msg = f"Epoch {self.current_epoch} Batch {batch+1}: loss={logs.get('loss'):.4f}"
        self.log_func(msg)
        QApplication.processEvents()
        
        

class BrainTumorTFModel:
    def __init__(self):
        self.model = None
        self.num_classes = 4
        self.class_indices = None
        self.class_labels = None

    def build_model(self, input_shape=(64, 64, 3)):
        self.model = tf.keras.Sequential([
            tf.keras.Input(shape=input_shape),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            Dropout(0.25),

            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            Dropout(0.25),

            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            Dropout(0.25),

            Flatten(),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax')
        ])
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self, data_dir, epochs=25, batch_size=32, log_func=print):
        if self.model is None:
            log_func("Building TensorFlow model...")
            self.build_model()

        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True
        )
        val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

        train_data = train_datagen.flow_from_directory(
            os.path.join(data_dir, "train"),
            target_size=(64, 64),
            batch_size=batch_size,
            class_mode='categorical'
        )
        val_data = val_datagen.flow_from_directory(
            os.path.join(data_dir, "val"),
            target_size=(64, 64),
            batch_size=batch_size,
            class_mode='categorical'
        )

        callback = QTextEditLogger(log_func)
        log_func("Starting TensorFlow training...")

        history = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            steps_per_epoch=len(train_data),
            validation_steps=len(val_data),
            callbacks=[callback],
            verbose=0
        )

        log_func("TensorFlow training complete.")
        # ✅ Switch to histogram tab
        if hasattr(self, 'show_histogram_tab'):
            self.show_histogram_tab()
        
        return history

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = load_model(path)

    def evaluate(self, data_dir, batch_size=32):
        test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

        test_data = test_datagen.flow_from_directory(
            os.path.join(data_dir, "test"),
            target_size=(64, 64),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )

        loss, acc = self.model.evaluate(test_data, verbose=0)
        print(f"Test Accuracy: {acc:.4f}")

        return acc

    def predict(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.resize(image, (64, 64))
        image = image.astype('float32') / 255.0
        image = np.expand_dims(image, axis=0)
        prediction = self.model.predict(image)[0]
        pred_class = np.argmax(prediction)
        pred_prob = prediction[pred_class]

        if self.class_labels:
            label = self.class_labels[pred_class]
            return label, pred_prob
        return pred_class, pred_prob
    

class BrainTumorPTModel(nn.Module):
    def __init__(self, num_classes=4):
        super(BrainTumorPTModel, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )

        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.dropout_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.dropout_fc(x)
        x = self.fc2(x)
        return x



class BrainTumorPTHandler:
    def __init__(self, num_classes=4, lr=0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BrainTumorPTModel(num_classes).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, data_dir, epochs=15, batch_size=32, log_func=print):
        if self.model is None:
            log_func("Rebuilding PyTorch model...")
            self.model = BrainTumorPTModel().to(self.device)
        
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        
        transform_train = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            normalize,
        ])
        transform_test = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            normalize,
        ])

        train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform_train)
        print("PyTorch Training class_to_idx:", train_dataset.class_to_idx)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=transform_test)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []

        self.model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            correct_train = 0
            total_train = 0
            
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                
                # Calculate training accuracy for the batch
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()
                
                # Log per batch loss
                log_func(f"Epoch {epoch+1} Batch {batch_idx+1}/{len(train_loader)} Loss: {loss.item():.4f}")
                
                # Process Qt events to keep UI responsive
                QApplication.processEvents()

            epoch_train_loss = running_loss / len(train_loader)
            epoch_train_acc = correct_train / total_train
            train_losses.append(epoch_train_loss)
            train_accuracies.append(epoch_train_acc)

            # Evaluate on validation set at end of epoch
            self.model.eval()
            running_val_loss = 0.0
            correct_val = 0
            total_val = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    running_val_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()

            epoch_val_loss = running_val_loss / len(val_loader)
            epoch_val_acc = correct_val / total_val
            val_losses.append(epoch_val_loss)
            val_accuracies.append(epoch_val_acc)

            log_func(f"Epoch {epoch+1} completed. Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, Val Acc: {epoch_val_acc:.4f}")

            self.model.train()  # Set back to train mode
            
            # ✅ Switch to histogram tab
            if hasattr(self, 'show_histogram_tab'):
                self.show_histogram_tab()

        # Return the recorded metrics for plotting later
        return train_losses, val_losses, train_accuracies, val_accuracies

    def evaluate(self, data_dir, batch_size=32, test_folder="test"):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            normalize,
        ])
        test_dataset = datasets.ImageFolder(os.path.join(data_dir, test_folder), transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print(f"Test Accuracy: {accuracy:.4f}")
        return accuracy

    def predict(self, image_path):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        image = cv2.resize(image, (64, 64))
        image = image.astype('float32') / 255.0
        
        # Convert numpy image (H,W,C) to tensor (C,H,W)
        image = np.transpose(image, (2, 0, 1))
        image = torch.tensor(image, dtype=torch.float32)
        
        # Apply normalization on tensor
        image = normalize(image)

        # Add batch dimension
        image = image.unsqueeze(0).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            output = self.model(image)
            probs = torch.softmax(output, dim=1)
            pred_prob, pred_class = torch.max(probs, dim=1)
            
        return pred_class.item(), pred_prob.item()

    def save(self, path='pt_model1.pth'):
        torch.save(self.model.state_dict(), path)

    def load(self, path='pt_model1.pth'):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

# ------------------- Dataset Tab ------------------------
class DatasetTab(QWidget):
    def __init__(self, dataset_type):
        super().__init__()
        self.dataset_type = dataset_type 
        self.init_models()
        self.initUI()

    def init_models(self):
        if self.dataset_type == "Brain Tumor":
            self.tf_model = BrainTumorTFModel()
            self.pt_model = BrainTumorPTHandler()
            self.class_names = {
                0: "glioma",
                1: "meningioma",
                2: "notumor",
                3: "pituitary"
            }

        self.dataset_dir = None
        self.image_path = None
        self.img_path = None


    def initUI(self):
        layout = QVBoxLayout()

        self.status_label = QLabel("Status: Ready")
        layout.addWidget(self.status_label)

        # Group 1: Load Dataset (blue)
        blue_style = """
            QPushButton {
            color: white;
            border-radius: 10px;
            padding: 6px 12px;
            border-style: solid;
            border-width: 2px;
            border-color: #4764b8 #365294 #365294 #4764b8;  /* top/left darker, bottom/right lighter */
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                        stop:0 #597cd0, stop:1 #3f5abf);
            font-weight: normal;
            font-size: 15px; 
            
        }
        QPushButton:hover {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                stop:0 #6a8ded, stop:1 #4d7adb);
        }
        QPushButton:pressed {
            padding-left: 8px;
            padding-top: 8px;
            border-color: #365294 #4764b8 #4764b8 #365294; /* invert border to simulate pressed */
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                        stop:0 #365294, stop:1 #2a4374);
        }
        """
        
        # Group 2: TensorFlow (purple)
        purple_style = """
            QPushButton {
                color: white;
                border-radius: 10px;
                padding: 6px 12px;
                border-style: solid;
                border-width: 2px;
                border-color: #a886b4 #8e6ea3 #8e6ea3 #a886b4;  /* softer border color */
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                            stop:0 #b89bcf, stop:1 #9d7cbf);  /* softer gradient */
                font-weight: normal;
                font-size: 15px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                            stop:0 #c3a0d8, stop:1 #a982c4);  /* lighter hover gradient */
            }
            QPushButton:pressed {
                padding-left: 8px;
                padding-top: 8px;
                border-color: #8e6ea3 #a886b4 #a886b4 #8e6ea3;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                            stop:0 #8e6ea3, stop:1 #6c4c80);  /* softer pressed gradient */
            }
        """

        # Group 3: Pytorch (pink)
        pink_style = """
            QPushButton {
                color: white;
                border-radius: 10px;
                padding: 6px 12px;
                border-style: solid;
                border-width: 2px;
                border-color: #ffb8e4 #f9a1d2 #f9a1d2 #ffb8e4;  /* softer border color */
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                            stop:0 #ffb8e4, stop:1 #f78ec6);  /* soft pink gradient */
                font-weight: normal;
                font-size: 15px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                            stop:0 #f9a1d2, stop:1 #f692bd);  /* lighter hover gradient */
            }
            QPushButton:pressed {
                padding-left: 8px;
                padding-top: 8px;
                border-color: #f9a1d2 #ffb8e4 #ffb8e4 #f9a1d2;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                            stop:0 #f9a1d2, stop:1 #d77f95);  /* softer pressed gradient */
            }
        """


        # Group 4: Info/Help (light blue)
        lightblue_style = """
            QPushButton {
                color: white;
                border-radius: 10px;
                padding: 6px 12px;
                border-style: solid;
                border-width: 2px;
                border-color: #8db5d7 #7ea1c9 #7ea1c9 #8db5d7;  /* softer border color */
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                            stop:0 #7ea1c9, stop:1 #4d8aab);  /* softer gradient */
                font-weight: normal;
                font-size: 15px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                            stop:0 #8eb9da, stop:1 #649dbd);  /* lighter hover gradient */
            }
            QPushButton:pressed {
                padding-left: 8px;
                padding-top: 8px;
                border-color: #7ea1c9 #8db5d7 #8db5d7 #7ea1c9;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                            stop:0 #4d8aab, stop:1 #3a6e8d);  /* softer pressed gradient */
            }
        """

        if self.dataset_type == "Brain Tumor":
            self.datasplit_btn = QPushButton("Split Brain Tumor Dataset")
            self.datasplit_btn.setStyleSheet(blue_style)
            self.datasplit_btn.clicked.connect(self.splitdataset)
            
        self.load_btn = QPushButton(f"Load {self.dataset_type} Dataset Folder")
        self.load_btn.setStyleSheet(blue_style)
        
        self.train_tf_btn = QPushButton("Train TensorFlow Model")
        self.train_tf_btn.setStyleSheet(purple_style)
        
        self.train_pt_btn = QPushButton("Train PyTorch Model")
        self.train_pt_btn.setStyleSheet(pink_style)
        
        self.show_tf_curve_btn = QPushButton("TensorFlow Training Curves")
        self.show_tf_curve_btn.setStyleSheet(purple_style)
        
        self.show_pt_curve_btn = QPushButton("PyTorch Training Curves")
        self.show_pt_curve_btn.setStyleSheet(pink_style)
        
        self.eval_tf_btn = QPushButton("Evaluate TensorFlow Model")
        self.eval_tf_btn.setStyleSheet(purple_style)
        
        self.eval_pt_btn = QPushButton("Evaluate PyTorch Model")
        self.eval_pt_btn.setStyleSheet(pink_style)
        
        self.load_img_detection2_btn = QPushButton("Tumor Object Detection")
        self.load_img_detection2_btn.setStyleSheet(blue_style)
         
        self.predict_tf_btn = QPushButton("Predict with TensorFlow")
        self.predict_tf_btn.setStyleSheet(purple_style)
        
        self.predict_pt_btn = QPushButton("Predict with PyTorch")
        self.predict_pt_btn.setStyleSheet(pink_style)
        
        self.metrics_btn = QPushButton("Show Performance Metrics")
        self.metrics_btn.setStyleSheet(blue_style)
        
        self.save_tf_model_btn = QPushButton("Save TensorFlow Model")
        self.save_tf_model_btn.setStyleSheet(purple_style)
        
        self.load_tf_model_btn = QPushButton("Load TensorFlow Model")
        self.load_tf_model_btn.setStyleSheet(purple_style)
        
        self.save_pt_model_btn = QPushButton("Save PyTorch Model")
        self.save_pt_model_btn.setStyleSheet(pink_style)
        
        self.load_pt_model_btn = QPushButton("Load PyTorch Model")
        self.load_pt_model_btn.setStyleSheet(pink_style)
        
        self.help_btn = QPushButton("Help / Info")
        self.help_btn.setStyleSheet(lightblue_style)
        
        # Group box for TensorFlow model buttons
        tf_group = QGroupBox("TensorFlow Model")
        tf_group.setStyleSheet("QGroupBox { color: white; font-weight: bold; font-size: 15px;}")
        tf_layout = QVBoxLayout()
        tf_layout.addWidget(self.train_tf_btn)
        tf_layout.addWidget(self.predict_tf_btn)
        tf_layout.addWidget(self.save_tf_model_btn)
        tf_layout.addWidget(self.load_tf_model_btn)
        tf_group.setLayout(tf_layout)
        
        # Group PyTorch buttons in a QGroupBox
        pt_group = QGroupBox("PyTorch Model")
        pt_group.setStyleSheet("QGroupBox { color: white; font-weight: bold; font-size: 15px;}")
        pt_layout = QVBoxLayout()
        pt_layout.addWidget(self.train_pt_btn)
        pt_layout.addWidget(self.predict_pt_btn)
        pt_layout.addWidget(self.save_pt_model_btn)
        pt_layout.addWidget(self.load_pt_model_btn)
        pt_group.setLayout(pt_layout)

        # Add all buttons to the main layout
        if hasattr(self, "datasplit_btn"):
            layout.addWidget(self.datasplit_btn)
            
        layout.addWidget(self.load_btn)

        layout.addWidget(tf_group)  
        layout.addWidget(pt_group) 
        layout.addWidget(self.help_btn)
        layout.addStretch()

        self.setLayout(layout)

        # Connect buttons
        self.load_btn.clicked.connect(self.load_dataset)
        self.train_tf_btn.clicked.connect(self.train_tf)
        self.train_pt_btn.clicked.connect(self.train_pt)
        self.eval_tf_btn.clicked.connect(self.evaluate_tf)
        self.eval_pt_btn.clicked.connect(self.evaluate_pt)
        self.show_tf_curve_btn.clicked.connect(lambda: self.show_training_curves_button_clicked('tf'))
        self.show_pt_curve_btn.clicked.connect(lambda: self.show_training_curves_button_clicked('pt'))
     
        self.help_btn.clicked.connect(self.show_help)
        self.predict_tf_btn.clicked.connect(self.load_image_tf)
        self.predict_pt_btn.clicked.connect(self.load_image_pt)
        self.metrics_btn.clicked.connect(self.show_performance_metrics)
        
        self.save_tf_model_btn.clicked.connect(self.save_tf_model)
        self.load_tf_model_btn.clicked.connect(self.load_tf_model)
        self.save_pt_model_btn.clicked.connect(self.save_pt_model)
        self.load_pt_model_btn.clicked.connect(self.load_pt_model)

    def log(self, message):
        print(message)
        
    # Brain Tumor dataset split
    def splitdataset(self):
        # Select original dataset folder
        src_folder = QFileDialog.getExistingDirectory(self, f"Select {self.dataset_type} Dataset Folder (with Training only)")
        if not src_folder:
            return

        # Select destination folder
        dst_base = QFileDialog.getExistingDirectory(self, "Select Destination to Create New Dataset")
        if not dst_base:
            return

        new_dataset_name = f"{self.dataset_type}_SplitDataset_7_2_1"
        dst_folder = os.path.join(dst_base, new_dataset_name)
        os.makedirs(dst_folder, exist_ok=True)
        self.log(f"Creating new dataset at: {dst_folder}")

        # Define destination subfolders
        dst_train = os.path.join(dst_folder, "train")
        dst_val = os.path.join(dst_folder, "val")
        dst_test = os.path.join(dst_folder, "test")

        os.makedirs(dst_train, exist_ok=True)
        os.makedirs(dst_val, exist_ok=True)
        os.makedirs(dst_test, exist_ok=True)

        src_train = os.path.join(src_folder, "Training")
        if not os.path.exists(src_train):
            self.log("❌ 'Training' folder not found.")
            return

        for cls in os.listdir(src_train):
            src_cls_path = os.path.join(src_train, cls)
            if not os.path.isdir(src_cls_path):
                continue

            images = [f for f in os.listdir(src_cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if len(images) < 5:
                self.log(f"⚠️ Skipping class '{cls}': not enough images.")
                continue

            # Step 1: split into train and temp (70% / 30%)
            train_imgs, temp_imgs = train_test_split(images, test_size=0.3, random_state=42)
            # Step 2: split temp into val and test (2:1 ratio → 20% / 10%)
            val_imgs, test_imgs = train_test_split(temp_imgs, test_size=1/3, random_state=42)

            # Prepare class dirs
            dst_cls_train = os.path.join(dst_train, cls)
            dst_cls_val = os.path.join(dst_val, cls)
            dst_cls_test = os.path.join(dst_test, cls)

            os.makedirs(dst_cls_train, exist_ok=True)
            os.makedirs(dst_cls_val, exist_ok=True)
            os.makedirs(dst_cls_test, exist_ok=True)

            # Copy images
            for img in train_imgs:
                os.link(os.path.join(src_cls_path, img), os.path.join(dst_cls_train, img))
            for img in val_imgs:
                os.link(os.path.join(src_cls_path, img), os.path.join(dst_cls_val, img))
            for img in test_imgs:
                os.link(os.path.join(src_cls_path, img), os.path.join(dst_cls_test, img))

        self.log("✅ Dataset successfully split into train (70%), val (20%), test (10%).")

    def log(self, message):
        print(message)    

    def load_dataset(self):
        folder = QFileDialog.getExistingDirectory(self, f"Select {self.dataset_type} Dataset Folder")
        if folder:
            self.dataset_dir = folder
            self.log(f"Dataset loaded: {folder}")

    def train_tf(self):
        if not self.dataset_dir:
            self.log("Load dataset folder first!")
            return
        try:
            self.log("Training TensorFlow model...")
            history = self.tf_model.train(self.dataset_dir, log_func=self.log)
            self.tf_history = history.history
            self.log("TensorFlow training complete.")
            
            # Show training curve
            self.show_training_curves('tf')

            # Evaluate and show confusion matrix
            self.evaluate_tf()
            self.compute_confusion_matrix('tf')
        except Exception as e:
            self.log(f"Error during TensorFlow training: {e}")

    def train_pt(self):
        if not self.dataset_dir:
            self.log("Load dataset folder first!")
            return
        self.log("Training PyTorch model...")
        train_losses, val_losses, train_accs, val_accs = self.pt_model.train(self.dataset_dir, log_func=self.log)
        self.pt_history = {
            'train_loss': train_losses,
            'val_loss': val_losses,
            'train_acc': train_accs,
            'val_acc': val_accs
        }
        self.log("PyTorch training complete.")
        
        # Show training curve
        self.show_training_curves('pt')

        # Evaluate and show confusion matrix
        self.evaluate_pt()
        self.compute_confusion_matrix('pt')

    def evaluate_tf(self):
        if not self.dataset_dir:
            self.log("Load dataset folder first!")
            return
        self.log("Evaluating TensorFlow model...")

        acc = self.tf_model.evaluate(self.dataset_dir)
        self.last_eval_model = 'tf'
        self.log(f"TensorFlow evaluation accuracy: {acc:.2f}")

    def evaluate_pt(self):
        if not self.dataset_dir:
            self.log("Load dataset folder first!")
            return

        self.log("Evaluating PyTorch model...")

        test_folder_name = "test" if self.dataset_type == "Brain Tumor" else "test"
        acc = self.pt_model.evaluate(self.dataset_dir, test_folder=test_folder_name)
        self.last_eval_model = 'pt'
        self.log(f"PyTorch evaluation accuracy: {acc:.2f}")

    def load_image_tf(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Load Image", "", "Images (*.png *.jpg *.bmp *.jpeg)")
        if file_name:
            self.log(f"Selected file: {file_name}")
            self.image_path = file_name
            self.img_path = file_name

            main_window = self.parentWidget()
            while main_window and not isinstance(main_window, QMainWindow):
                main_window = main_window.parentWidget()

            if main_window and hasattr(main_window, 'image_label') and hasattr(main_window, 'object_detection_label'):
                self.image_label = main_window.image_label
                self.object_detection_label = main_window.object_detection_label

                pixmap = QPixmap(file_name)
                if not pixmap.isNull():
                    self.log("Pixmap loaded successfully.")

                    # Only set image_label (left panel)
                    scaled_pixmap_image = pixmap.scaled(
                        self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    self.image_label.setPixmap(scaled_pixmap_image)

                    # Predict and auto-detect (object_detection sets object_detection_label)
                    self.predict_tf()

                    # Automatically run detection
                    if self.dataset_type == "Brain Tumor":
                        self.detect_tumor()

                else:
                    self.log("Failed to load image preview: pixmap is null.")

                    
    def predict_tf(self):
        if not hasattr(self, 'image_path') or not self.image_path:
            self.log("No image loaded for detection.")
            return
        
        image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            self.log("Failed to read image from path.")
            return
        pred_class, pred_prob = self.tf_model.predict(self.image_path)
        label = self.class_names.get(pred_class, f"Class {pred_class}")
        self.log(f"TensorFlow Prediction: {label} ({pred_prob*100:.2f}%)")

                    
    def load_image_pt(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Load Image", "", "Images (*.png *.jpg *.bmp *.jpeg)")
        if file_name:
            self.log(f"Selected file: {file_name}")
            self.image_path = file_name
            self.img_path = file_name  

            # Get reference to main window
            main_window = self.parentWidget()
            while main_window and not isinstance(main_window, QMainWindow):
                main_window = main_window.parentWidget()

            if main_window and hasattr(main_window, 'image_label') and hasattr(main_window, 'object_detection_label'):
                self.image_label = main_window.image_label
                self.object_detection_label = main_window.object_detection_label

                # Load pixmap and scale for both image panels
                pixmap = QPixmap(file_name)
                if not pixmap.isNull():
                    self.log("Pixmap loaded successfully.")

                    scaled_pixmap_image = pixmap.scaled(
                        self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    scaled_pixmap_detect = pixmap.scaled(
                        self.object_detection_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

                    self.image_label.setPixmap(scaled_pixmap_image)
                    self.object_detection_label.setPixmap(scaled_pixmap_detect)

                    # Predict class using PyTorch model
                    self.predict_pt()

                    # Automatically trigger detection based on dataset type
                    if self.dataset_type == "Brain Tumor":
                        self.detect_tumor()
    
                else:
                    self.log("Failed to load image preview: pixmap is null.")

                    
    def predict_pt(self):
        if not hasattr(self, 'image_path') or not self.image_path:
            self.log("No image loaded for detection.")
            return
        
        image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            self.log("Failed to read image from path.")
            return
        
        try:
            pred_class, pred_prob = self.pt_model.predict(self.image_path)
            label = self.class_names.get(pred_class, f"Class {pred_class}")
            self.log(f"PyTorch Prediction: {label} ({pred_prob*100:.2f}%)")
        except Exception as e:
            self.log(f"Prediction failed: {e}")

    def object_detection(self):
        if not hasattr(self, 'img_path') or not self.img_path:
            self.log("No image loaded for detection.")
            return

        image = cv2.imread(self.img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            self.log("Failed to read image from path.")
            return

        _, thresh_image = cv2.threshold(image, 180, 255, cv2.THRESH_BINARY)
        kernel = np.ones((7, 7), np.uint8)
        morph_image = cv2.morphologyEx(thresh_image, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(morph_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        for contour in contours:
            if cv2.contourArea(contour) > 1500:
                area = cv2.contourArea(contour)
                color = (0, 255, 255) if area < 5000 else (0, 255, 0) if area < 10000 else (0, 0, 255)
                cv2.drawContours(output_image, [contour], -1, color, 2)

        self.display_image_with_bounding_boxes(output_image)

    def display_image_with_bounding_boxes(self, image):
        """Display the image with color-coded contours for fluid."""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channel = rgb_image.shape
        bytes_per_line = 3 * width
        q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap(q_image)
        self.object_detection_label.setPixmap(pixmap.scaled(
            self.object_detection_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        ))

    def load_image_tumor(self):
        """Load an image using a file dialog and automatically apply object detection."""
        try:
            file_name, _ = QFileDialog.getOpenFileName(self, "Load Image", "", "Images (*.png *.jpg *.bmp *.jpeg)")
            if file_name:
                self.img_path = file_name

                # Find the main window and access the label
                main_window = self.parentWidget()
                while main_window and not isinstance(main_window, QMainWindow):
                    main_window = main_window.parentWidget()

                if main_window and hasattr(main_window, 'object_detection_label'):
                    self.object_detection_label = main_window.object_detection_label
                    pixmap = QPixmap(file_name)
                    if not pixmap.isNull():
                        pixmap = pixmap.scaled(self.object_detection_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                        self.object_detection_label.setPixmap(pixmap)

                        self.detect_tumor()
                    else:
                        self.log("Failed to load image preview: QPixmap is null.")
                else:
                    self.log("Main window or object_detection_label not found.")
        except Exception as e:
            import traceback
            self.log(f"Unhandled exception in load_image_tumor: {e}")
            self.log(traceback.format_exc())
             
    def detect_tumor(self):
        try:
            if not hasattr(self, 'img_path') or not self.img_path:
                self.log("No image loaded for detection.")
                return

            gray_image = cv2.imread(self.img_path, cv2.IMREAD_GRAYSCALE)
            if gray_image is None:
                self.log("Failed to read image from path.")
                return

            original_bgr = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

            # Define thresholds
            lower_1, upper_1 = 85, 170
            upper_2 = 255

            dark_grey = cv2.inRange(gray_image, 0, lower_1)
            light_grey = cv2.inRange(gray_image, lower_1 + 1, upper_1)
            white = cv2.inRange(gray_image, upper_1 + 1, upper_2)

            # Create blank color mask
            mask = np.zeros_like(original_bgr)
            mask[dark_grey == 255] = [255, 0, 0]     # Blue
            mask[light_grey == 255] = [0, 255, 0]    # Green
            mask[white == 255] = [0, 0, 255]         # Red

            # Blend with original
            blended = cv2.addWeighted(original_bgr, 0.7, mask, 0.3, 0)

            self.display_image_from_array(blended)
        except Exception as e:
            import traceback
            self.log(f"Unhandled exception in detect_tumor: {e}")
            self.log(traceback.format_exc())


    def display_image_from_array(self, image_array):
        """Display an image given a numpy array."""
        height, width, channel = image_array.shape
        bytes_per_line = 3 * width
        q_image = QImage(image_array.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap(q_image.rgbSwapped())
        self.object_detection_label.setPixmap(pixmap.scaled(self.object_detection_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def show_help(self):
        info_text = (
            f"<b> User Manual </b><br><br>"
            "Train model<br><br>"
            "1. Split the original dataset.<br>"
            "***Note: Dataset folder should contain exact folder name with 'train' and 'test' subfolders<br> "
            "2. Load the splitted dataset.<br>"
            "3. Train TensorFlow or PyTorch models using respective buttons.<br>"
            "4. After finish training model, the evaluation results such as test accuracy, confusion matrix and training curve will show in the respective tabs.<br>"
            "***Note: if the accuracy is not over 0.6, feel free to click train button again to run the training process second time to achieve higher accuracy.***<br>"
            "5. Save model for future use.<br><br><br>"
            "Prediction Step<br><br>"
            "1. Load the splitted dataset first then load a model.<br>"
            "2. Click predict button and pick an image.<br>"
            "3. Image classification will be shown in the bottom part of output console with similarity percentages.<br><br><br>"
        )
        QMessageBox.information(self, f"{self.dataset_type} Help", info_text)                
        
    def show_performance_metrics(self):
        if not self.dataset_dir:
            self.log("Load dataset folder first!")
            return

        self.log("Computing performance metrics (confusion matrix)...")

        if hasattr(self, 'last_eval_model'):
            self.compute_confusion_matrix(model=self.last_eval_model)
        
        
    def compute_confusion_matrix(self, model='tf'):
        self.log(f"Computing confusion matrix for {'TensorFlow' if model == 'tf' else 'PyTorch'} model...")

        test_folder = "test" if self.dataset_type == "Brain Tumor" else "test"

        try:
            if model == 'tf':
                datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
                tf_test = datagen.flow_from_directory(
                    os.path.join(self.dataset_dir, test_folder),
                    target_size=(64, 64),
                    batch_size=32,
                    class_mode='categorical',
                    shuffle=False
                )

                tf_preds = self.tf_model.model.predict(tf_test)
                tf_pred_classes = np.argmax(tf_preds, axis=1)
                tf_true_classes = tf_test.classes

                cm = confusion_matrix(tf_true_classes, tf_pred_classes)
                title = "TensorFlow Confusion Matrix"
                classes = list(tf_test.class_indices.keys())

            elif model == 'pt':
                normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor(), normalize])
                pt_test = datasets.ImageFolder(os.path.join(self.dataset_dir, test_folder), transform=transform)
                pt_loader = DataLoader(pt_test, batch_size=32, shuffle=False)

                y_true, y_pred = [], []
                self.pt_model.model.eval()
                with torch.no_grad():
                    for imgs, labels in pt_loader:
                        imgs = imgs.to(self.pt_model.device)
                        outputs = self.pt_model.model(imgs)
                        _, preds = torch.max(outputs, 1)
                        y_true.extend(labels.numpy())
                        y_pred.extend(preds.cpu().numpy())

                cm = confusion_matrix(y_true, y_pred)
                title = "PyTorch Confusion Matrix"
                classes = pt_test.classes

            else:
                self.log("Invalid model specified.")
                return

            # Plotting
            fig, ax = plt.subplots()
            im = ax.imshow(cm, cmap='Blues')
            plt.colorbar(im)
            ax.set_title(title)
            ax.set_xticks(np.arange(len(classes)))
            ax.set_yticks(np.arange(len(classes)))
            ax.set_xticklabels(classes, rotation=45)
            ax.set_yticklabels(classes)
            ax.set_ylabel('True label')
            ax.set_xlabel('Predicted label')

            for i in range(len(cm)):
                for j in range(len(cm[0])):
                    ax.text(j, i, cm[i, j], ha='center', va='center',
                            color='white' if cm[i, j] > cm.max() / 2. else 'black')

            buf = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format='png')
            plt.close(fig)
            buf.seek(0)

            if hasattr(self, 'show_histogram_callback'):
                self.show_histogram_callback(buf.read())
                self.log(f"{title} displayed.")
            else:
                self.log("No histogram display callback set.")

        except Exception as e:
            import traceback
            self.log(f"Error during confusion matrix computation: {e}")
            self.log(traceback.format_exc())

        
    def set_histogram_callback(self, callback):
        self.show_histogram_callback = callback
        
    def show_training_curves(self, framework='tf'):
        plt.figure(figsize=(10,4))
        epochs = None

        if framework == 'tf' and hasattr(self, 'tf_history'):
            history = self.tf_history
            epochs = range(1, len(history['loss']) + 1)
            plt.subplot(1,2,1)
            plt.plot(epochs, history['loss'], label='Training Loss')
            plt.plot(epochs, history['val_loss'], label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('TF Loss')
            plt.legend()

            plt.subplot(1,2,2)
            plt.plot(epochs, history['accuracy'], label='Training Accuracy')
            plt.plot(epochs, history['val_accuracy'], label='Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('TF Accuracy')
            plt.legend()

        elif framework == 'pt' and hasattr(self, 'pt_history'):
            history = self.pt_history
            epochs = range(1, len(history['train_loss']) + 1)
            plt.subplot(1,2,1)
            plt.plot(epochs, history['train_loss'], label='Training Loss')
            plt.plot(epochs, history['val_loss'], label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('PT Loss')
            plt.legend()

            plt.subplot(1,2,2)
            plt.plot(epochs, history['train_acc'], label='Training Accuracy')
            plt.plot(epochs, history['val_acc'], label='Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('PT Accuracy')
            plt.legend()
        else:
            self.log(f"No training history available for {framework}.")
            return

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        img_bytes = buf.read()
        
        # Emit to the main window's callback to show in new tab
        if hasattr(self, 'show_training_callback'):
            self.show_training_callback(img_bytes)
            self.log("Training curves displayed.")
        else:
            self.log("No callback set for training curves display.")    
            
    def set_training_callback(self, callback):
        self.show_training_callback = callback
        
    def show_training_curves_button_clicked(self, framework):
        if framework == 'tf' and hasattr(self, 'tf_history'):
            self.show_training_curves(framework='tf')
        elif framework == 'pt' and hasattr(self, 'pt_history'):
            self.show_training_curves(framework='pt')
        else:
            self.log(f"No training history available for {framework}.")
            return

        main_window = self.parentWidget()
        while main_window and not isinstance(main_window, QMainWindow):
            main_window = main_window.parentWidget()

        if main_window and hasattr(main_window, 'right_tabs'):
            for i in range(main_window.right_tabs.count()):
                if main_window.right_tabs.tabText(i).lower() == "training curves":
                    main_window.right_tabs.setCurrentIndex(i)
                    break
    
    def save_training_history(self, path):
        """Save training history dict as JSON file."""
        if not hasattr(self, 'tf_history') and not hasattr(self, 'pt_history'):
            self.log("No training history to save.")
            return
        history_data = {}
        if hasattr(self, 'tf_history'):
            history_data['tf_history'] = self.tf_history
        if hasattr(self, 'pt_history'):
            history_data['pt_history'] = self.pt_history
        try:
            with open(path, 'w') as f:
                json.dump(history_data, f)
            self.log(f"Training history saved to {path}")
        except Exception as e:
            self.log(f"Failed to save training history: {e}")

    def load_training_history(self, path):
        """Load training history JSON file."""
        try:
            with open(path, 'r') as f:
                history_data = json.load(f)
            if 'tf_history' in history_data:
                self.tf_history = history_data['tf_history']
                self.log(f"TensorFlow training history loaded from {path}")
            if 'pt_history' in history_data:
                self.pt_history = history_data['pt_history']
                self.log(f"PyTorch training history loaded from {path}")
        except Exception as e:
            self.log(f"Failed to load training history: {e}")

    # Save TensorFlow model
    def save_tf_model(self):
        if not self.dataset_dir:
            self.log("Train or load a TensorFlow model first!")
            return
        filename, _ = QFileDialog.getSaveFileName(self, "Save TensorFlow Model", "", "TensorFlow Model Files (*.h5)")
        if filename:
            if not filename.endswith('.h5'):
                self.log("Unsupported file extension for TensorFlow model. Use .h5")
                return
            self.tf_model.save(filename)
            self.log(f"TensorFlow model saved to {filename}")
            # Save training history JSON alongside model
            history_path = filename + ".history.json"
            self.save_training_history(history_path)

    # Load TensorFlow model                
    def load_tf_model(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Load TensorFlow Model", "", "TensorFlow Model Files (*.h5)")
        if filename:
            try:
                self.tf_model.load(filename)
                self.log(f"TensorFlow model loaded from {filename}")
                # Try loading history JSON too
                history_path = filename + ".history.json"
                if os.path.exists(history_path):
                    self.load_training_history(history_path)
                    self.show_training_curves('tf')
                else:
                    self.log("No training history file found for this model.")
                
                # Evaluate and show confusion matrix
                self.evaluate_tf()
                self.compute_confusion_matrix('tf')
                
                # ✅ Switch to histogram tab
                if hasattr(self, 'show_histogram_tab'):
                    self.show_histogram_tab()
            
            except Exception as e:
                self.log(f"Failed to load TensorFlow model: {e}")
          
    # Save PyTorch model            
    def save_pt_model(self):
        if not self.dataset_dir:
            self.log("Train or load a PyTorch model first!")
            return
        filename, _ = QFileDialog.getSaveFileName(self, "Save PyTorch Model", "", "PyTorch Model Files (*.pth)")
        if filename:
            if not filename.endswith('.pth'):
                self.log("Unsupported file extension for PyTorch model. Use .pth")
                return
            self.pt_model.save(filename)
            self.log(f"PyTorch model saved to {filename}")
            # Save training history JSON alongside model
            history_path = filename + ".history.json"
            self.save_training_history(history_path)

    # Load PyTorch model
    def load_pt_model(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Load PyTorch Model", "", "PyTorch Model Files (*.pth)")
        if filename:
            try:
                self.pt_model.load(filename)
                self.log(f"PyTorch model loaded from {filename}")
                # Try loading history JSON too
                history_path = filename + ".history.json"
                if os.path.exists(history_path):
                    self.load_training_history(history_path)
                    self.show_training_curves('pt')
                else:
                    self.log("No training history file found for this model.")
                 # Evaluate and show confusion matrix
                self.evaluate_pt()
                self.compute_confusion_matrix('pt')
                
                # ✅ Switch to histogram tab
                if hasattr(self, 'show_histogram_tab'):
                    self.show_histogram_tab()
            
            except Exception as e:
                self.log(f"Failed to load PyTorch model: {e}")
                
# ---------------- Emitting Stream for Console Output ------------------

class EmittingStream:
    def __init__(self, text_edit):
        self.text_edit = text_edit

    def write(self, text):
        if text.strip() != "":
            # Append text to QTextEdit (in the GUI thread)
            self.text_edit.append(text)

    def flush(self):
        pass  # Required for file-like object compatibility 
    
# ---------------- Main Window ------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CNN Image Classifier & Object Detector")
        self.setGeometry(100, 100, 900, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        self.central_widget.setStyleSheet("color: white; background-color: #1E1E1E;")

        main_layout = QHBoxLayout()

        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #444444;
                background: #1E1E1E;
            }
            QTabBar::tab {
                background: #1E1E1E;
                color: white;
                padding: 8px;
                border: 1px solid #444444;
                border-bottom-color: #1E1E1E;
                min-width: 120px;
            }
            QTabBar::tab:selected {
                background: #2D2D2D;
                border-color: #777777;
                border-bottom-color: #2D2D2D;
            }
            QTabBar::tab:hover {
                background: #3A3A3A;
            }
        """)

        self.braintumor_tab = DatasetTab("Brain Tumor")

        self.braintumor_tab.set_histogram_callback(self.show_histogram_tab)
        
        self.braintumor_tab.set_training_callback(self.show_training_curves_tab)

        self.tabs.addTab(self.braintumor_tab, "Brain Tumor")
    
        # Right side: image preview + console log
        right_layout = QVBoxLayout()
        
        # Create a container widget for the two image preview boxes
        self.image_preview_container = QWidget()
        image_preview_layout = QHBoxLayout()
        self.image_preview_container.setLayout(image_preview_layout)
        
        # Left image preview box
        self.image_label = QLabel("Image Preview")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedHeight(400)
        self.image_label.setStyleSheet("border: 1px solid gray; background-color: black; color: white;")

        # Right image preview label (object detection result)
        self.object_detection_label = QLabel("Object Detection Preview")
        self.object_detection_label.setAlignment(Qt.AlignCenter)
        self.object_detection_label.setFixedHeight(400)
        self.object_detection_label.setStyleSheet("border: 1px solid gray; background-color: black; color: white;")

        # Add both labels to the horizontal layout container
        image_preview_layout.addWidget(self.image_label)
        image_preview_layout.addWidget(self.object_detection_label)

        # Add the container with both previews to the right layout
        right_layout.addWidget(self.image_preview_container)

        # tabs widget for output console and histograms
        self.right_tabs = QTabWidget()
        self.right_tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #444444;
                background: #121212;
            }
            QTabBar::tab {
                background: #121212;
                color: white;
                padding: 6px;
                border: 1px solid #444444;
                border-bottom-color: #121212;
                min-width: 100px;
            }
            QTabBar::tab:selected {
                background: #2D2D2D;
                border-color: #777777;
                border-bottom-color: #2D2D2D;
            }
            QTabBar::tab:hover {
                background: #333333;
            }
        """)
        
        # Console tab
        self.output_console = QTextEdit()
        self.output_console.setReadOnly(True)
        self.right_tabs.addTab(self.output_console, "Output Console")
        self.output_console.setStyleSheet("background-color: black; border: 1px solid gray;")
        
        sys.stdout = EmittingStream(self.output_console)
        sys.stderr = EmittingStream(self.output_console)

        # Histogram tab 
        self.histogram_label = QLabel("Histograms will appear here")
        self.histogram_label.setAlignment(Qt.AlignCenter)
        self.histogram_label.setStyleSheet("background-color: white; border: 1px solid gray;")
        self.right_tabs.addTab(self.histogram_label, "Histograms")
        
        # Training curves tab
        self.training_curves_label = QLabel("Training curves will appear here")
        self.training_curves_label.setAlignment(Qt.AlignCenter)
        self.training_curves_label.setStyleSheet("background-color: black; border: 1px solid gray;")
        self.right_tabs.addTab(self.training_curves_label, "Training Curves")

        right_layout.addWidget(self.right_tabs)

        main_layout.addWidget(self.tabs, 1)
        main_layout.addLayout(right_layout, 3)

        self.central_widget.setLayout(main_layout)

        # Connect dataset tabs' log function to update the shared output console
        self.braintumor_tab.log = self.log

    def log(self, message):
        self.output_console.append(message)
        self.right_tabs.setCurrentWidget(self.output_console)
        
    def show_histogram_tab(self, image_data):
        # Load QPixmap from bytes image data
        pixmap = QPixmap()
        pixmap.loadFromData(image_data)
        pixmap = pixmap.scaled(self.histogram_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.histogram_label.setPixmap(pixmap)
        # Switch to histogram tab
        self.right_tabs.setCurrentWidget(self.histogram_label)
        
    def show_training_curves_tab(self, image_data):
        pixmap = QPixmap()
        pixmap.loadFromData(image_data)
        pixmap = pixmap.scaled(self.training_curves_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.training_curves_label.setPixmap(pixmap)
        self.right_tabs.setCurrentWidget(self.training_curves_label)
        
    def closeEvent(self, event):
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        event.accept()
    
if __name__ == '__main__':
    import io
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())