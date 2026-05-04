from engine.loop import run_training_experiment

def train_model(model, train_loader, val_loader, optimizer, criterion, device, config, save_path):

    return run_training_experiment(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=config["epochs"],
        save_path=save_path,
        patience=config.get("patience", 5)
    )