# Classifier agent owns fine-tuning and parameter tuning

Previously, fine-tuning lived in a separate `FinetuningAgent`, and Manager (`jack_manager.py::_approved_training_params`) computed the next learning rate and focus weight from iteration history. We moved both the training engine's ownership *and* the hyperparameter-selection intelligence into Nadi (the classifier agent): `FinetuningAgent` is retired, Nadi calls `train_finbert()` directly and picks its own learning rate / focus weight per head from its own history.

Manager keeps only the proceed-vs-retune gate (stop/continue, reading Sabina's report and score history) — it no longer decides *how* to fix the model, only *whether* another attempt is warranted. This crosses the original "stay in your lane, one agent one file" boundary deliberately: the alternative (Manager still computing hyperparameters, Nadi just executing them) would have relabeled the ownership without actually moving the tuning intelligence.
