import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from dataset import SignLanguageDataset
from model import Sign3DCNN


def main():
    torch.backends.cudnn.benchmark = True

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {DEVICE}")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    CSV_PATH = os.path.join(BASE_DIR, "annotations1.csv")
    TRAIN_DIR = os.path.join(BASE_DIR, "train")
    TEST_DIR = os.path.join(BASE_DIR, "test")
    MODEL_PATH = os.path.join(BASE_DIR, "model_3dcnn.pth")

    # Ускоренные параметры
    BATCH_SIZE = 8
    EPOCHS = 15
    LEARNING_RATE = 1e-4
    SEQUENCE_LENGTH = 8
    IMG_SIZE = 96
    NUM_WORKERS = 0
    EARLY_STOPPING_PATIENCE = 5

    train_dataset = SignLanguageDataset(
        CSV_PATH,
        TRAIN_DIR,
        sequence_length=SEQUENCE_LENGTH,
        img_size=IMG_SIZE,
        train=True,
        verbose=True,
        use_cache=True,
        augment=True
    )

    test_dataset = SignLanguageDataset(
        CSV_PATH,
        TEST_DIR,
        sequence_length=SEQUENCE_LENGTH,
        img_size=IMG_SIZE,
        train=False,
        verbose=True,
        use_cache=True,
        augment=False
    )

    # Оставляем все общие буквы между train и test
    train_classes = set(train_dataset.data["text"].unique())
    test_classes = set(test_dataset.data["text"].unique())
    selected_classes = sorted(train_classes.intersection(test_classes))

    if len(selected_classes) == 0:
        raise ValueError("Не найдено общих букв между train и test.")

    if len(selected_classes) != 33:
        raise ValueError(
            f"Ожидалось 33 буквы русского алфавита, но найдено {len(selected_classes)} классов: {selected_classes}"
        )

    print(f"Количество букв для обучения: {len(selected_classes)}")
    print("Буквы:", selected_classes)

    train_dataset.data = train_dataset.data[
        train_dataset.data["text"].isin(selected_classes)
    ].reset_index(drop=True)

    test_dataset.data = test_dataset.data[
        test_dataset.data["text"].isin(selected_classes)
    ].reset_index(drop=True)

    class_to_idx = {c: i for i, c in enumerate(selected_classes)}
    train_dataset.classes = selected_classes
    train_dataset.class_to_idx = class_to_idx
    test_dataset.classes = selected_classes
    test_dataset.class_to_idx = class_to_idx

    print(f"Размер train_dataset после фильтрации: {len(train_dataset)}")
    print(f"Размер test_dataset после фильтрации: {len(test_dataset)}")

    train_counts = train_dataset.data["text"].value_counts()
    class_weights = []

    for cls in selected_classes:
        count = train_counts.get(cls, 1)
        class_weights.append(1.0 / count)

    class_weights = np.array(class_weights, dtype=np.float32)
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

    print("Веса классов:", class_weights.cpu().numpy())

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False
    )

    num_classes = len(train_dataset.classes)
    model = Sign3DCNN(num_classes=num_classes).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    best_acc = 0.0
    epochs_without_improvement = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        train_preds = []
        train_labels = []

        for videos, labels in train_loader:
            videos = videos.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                outputs = model(videos)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            train_preds.extend(preds.detach().cpu().numpy())
            train_labels.extend(labels.detach().cpu().numpy())

        avg_loss = total_loss / max(len(train_loader), 1)
        train_acc = accuracy_score(train_labels, train_preds) if train_labels else 0.0

        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for videos, labels in test_loader:
                videos = videos.to(DEVICE, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                    outputs = model(videos)

                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        acc = accuracy_score(all_labels, all_preds) if all_labels else 0.0
        scheduler.step(acc)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Loss: {avg_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Acc: {acc:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        if acc > best_acc:
            best_acc = acc
            epochs_without_improvement = 0

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "classes": train_dataset.classes,
                "sequence_length": SEQUENCE_LENGTH,
                "img_size": IMG_SIZE,
                "num_classes": num_classes
            }
            torch.save(checkpoint, MODEL_PATH)
            print(f"Лучшая модель сохранена: {MODEL_PATH} | Acc={best_acc:.4f}")
        else:
            epochs_without_improvement += 1
            print(f"Без улучшения: {epochs_without_improvement}/{EARLY_STOPPING_PATIENCE}")

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print("Ранняя остановка: модель перестала улучшаться.")
            break

    print(f"\nЛучшая точность на test: {best_acc:.4f}")


if __name__ == "__main__":
    main()