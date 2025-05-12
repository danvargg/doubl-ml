import os

from fitml.data import DataLoader
from fitml.model import Predictor


def main(
        images_dir: str = "data/images",
        user_meta: str = "data/users.json",
        garment_meta: str = "data/garment.json",
        batch_size: int = 2,
        epochs: int = 10,
        output_dir: str = "outputs"
) -> None:
    data_loader = DataLoader(
        images_dir=images_dir,
        user_meta_path=user_meta,
        garment_meta_path=garment_meta,
        batch_size=batch_size,
        shuffle=True
    )

    print("generating dataset...")

    dataset = data_loader.get_dataset()

    for x_batch, y_batch in dataset.take(1):
        print("Sample batch shape:", x_batch.shape)
        print("Sample labels:", y_batch.numpy())
        input_dim = x_batch.shape[-1]

    print("building model...")
    predictor = Predictor(input_dim=input_dim)
    predictor.compile(learning_rate=lr)

    print("training model...")
    predictor.fit(dataset, epochs=epochs, verbose=1)

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "predictor_tf")
    predictor.model.save(save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
