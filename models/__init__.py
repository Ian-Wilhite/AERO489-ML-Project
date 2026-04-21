"""
models/ — Wing g-limit prediction model implementations.

Each module exposes one class that inherits from WingModel (base.py).

    Model 1 — LinearReg        (linear_reg.py)
    Model 2 — PolyReg          (poly_reg.py)
    Model 3 — GPR              (gpr.py)
    Model 4 — RandomForest     (random_forest.py)
    Model 5 — FeedforwardNN    (feedforward_nn.py)
    Model 6 — PINN             (pinn.py)
    Model 7 — DeepLearning     (deep_learning.py)

All models follow the WingModel interface:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = model.evaluate(X_test, y_test, n_features)
"""
