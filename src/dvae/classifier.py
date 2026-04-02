import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin
import scipy.sparse as ss

from dvae.vae import VAE
from dvae.config.hyperparameters import MINIBATCH_SIZE, N_EPOCHS_EVAL
from dvae.utils.logger import logger

USE_CUDA = torch.cuda.is_available()


class VAEClassifier(BaseEstimator, ClassifierMixin):
    """
    A full sklearn‑style classifier wrapper around the VAE.
    Replaces MyDataset, create_loader, train, test, classify_data.
    """

    # -------------------------
    # Internal Dataset
    # -------------------------
    class _Dataset(Dataset):
        def __init__(self, X, y=None):
            # Convert sparse → dense
            if isinstance(X, ss.spmatrix):
                X = X.toarray()

            X = np.asarray(X)

            self.encoder = torch.tensor(X, dtype=torch.float32)

            if y is not None:
                y = np.asarray(y)
                # Convert boolean labels to integer dtype to avoid torch dtype inference errors
                if y.dtype == np.bool_:
                    y = y.astype(np.int64)
                self.labels = torch.tensor(y, dtype=torch.long)
            else:
                self.labels = None

        def __len__(self):
            return len(self.encoder)

        def __getitem__(self, idx):
            d = {"encoder": self.encoder[idx]}
            if self.labels is not None:
                d["labels"] = self.labels[idx]
            return d

    # -------------------------
    # Constructor
    # -------------------------
    def __init__(self, feature_dim=None, **vae_kwargs):
        self.feature_dim = feature_dim
        self.vae_kwargs = vae_kwargs
        self.vae = None
        self.threshold_ = None
        self.is_fitted = False

    @classmethod
    def from_hparams(cls, h):
        return cls(
            feature_dim=h.input_dim,
            alpha1=h.alpha1,
            alpha2=h.alpha2,
            beta=h.beta,
            corruption=h.bcp,
            z_dim=h.z_dim,
            hidden_dim=h.hidden_dim,
            seed=h.seed,
        )

    # -------------------------
    # Loader factory
    # -------------------------
    def _loader(self, X, y=None, shuffle=False):
        ds = self._Dataset(X, y)
        return DataLoader(
            ds,
            batch_size=MINIBATCH_SIZE,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=USE_CUDA,
        )

    # -------------------------
    # Fit
    # -------------------------
    def fit(self, X, y, num_epochs=N_EPOCHS_EVAL):
        """
        Fit the VAE classifier.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,)
            Target labels.
        num_epochs : int, optional (default=N_EPOCHS_EVAL)
            Number of training epochs.
        Returns
        -------
        self : object
            Returns self.
        """

        X, y = np.asarray(X), np.asarray(y)

        # sklearn compatibility: store classes_ and input feature count
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]

        # Determine feature dimension if not provided
        if self.feature_dim is None:
            self.feature_dim = X.shape[1]

        # Instantiate VAE
        self.vae = VAE(self.feature_dim, **self.vae_kwargs)

        loader = self._loader(X, y, shuffle=False)
        score_train = self.vae.infer_parameters(loader, num_epochs=num_epochs)

        # Compute optimal threshold
        self.threshold_ = self._propose_threshold(score_train, y)

        self.is_fitted = True
        return self

    # -------------------------
    # Threshold selection
    # -------------------------
    def _propose_threshold(self, score_train, labels_train):
        """
        Propose threshold based on training scores and labels.

        Parameters
        ----------
        score_train : array-like, shape (n_samples,)
            Prediction scores in [0, 1].
        labels_train : array-like, shape (n_samples,)
            True labels.
        Returns
        -------
        th : float
            Optimal threshold.
        """
        thresholds = np.arange(0.05, 0.95, 0.05)
        accs = np.array(
            [accuracy_score(labels_train, score_train >= th) for th in thresholds]
        )
        best = np.max(accs)
        best_idxs = np.where(accs == best)[0]

        if len(best_idxs) > 1:
            logger.warning(f"Multiple best accuracies: {accs}")
            th = thresholds[best_idxs].mean()
        else:
            th = thresholds[best_idxs[0]]

        logger.debug(f"Chosen threshold: {th}")
        return th

    # -------------------------
    # Predict probabilities
    # -------------------------
    def predict_proba(self, X):
        """
        Predict probabilities for input data X.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data.
        Returns
        -------
        probs : array-like, shape (n_samples, n_classes)
            Predicted class probabilities (columns correspond to classes in `self.classes_`).
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit(...) before predict_proba(...).")

        loader = self._loader(X, None, shuffle=False)
        scores = self.vae.calc_predVI(loader)
        p_pos = np.asarray(scores)
        p_neg = 1.0 - p_pos
        return np.vstack([p_neg, p_pos]).T

    # -------------------------
    # Predict labels
    # -------------------------
    def predict(self, X):
        """
        Predict binary labels for input data X.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data.
        Returns
        -------
        labels : array-like, shape (n_samples,)
            Predicted binary labels.
        """
        probs = self.predict_proba(X)
        # Binary classifier assumption: exactly two classes
        if not hasattr(self, "classes_"):
            raise RuntimeError("Call fit(...) before predict(...).")
        if len(self.classes_) != 2:
            raise RuntimeError(
                "VAEClassifier supports only binary classification (2 classes)."
            )

        # Use fitted threshold if present, otherwise default to 0.5
        if not hasattr(self, "threshold_") or self.threshold_ is None:
            logger.warning("No threshold_ found; using default 0.5")
            th = 0.5
        else:
            th = self.threshold_

        # Positive-class probability is in column 1
        p_pos = probs[:, 1]
        mask = p_pos >= th
        return np.where(mask, self.classes_[1], self.classes_[0])

    # -------------------------
    # Transform (embeddings + reconstructions)
    # -------------------------
    def transform(self, X):
        """
        Transform input data X to obtain embeddings and reconstructions.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data.
        Returns
        -------
        scores : array-like, shape (n_samples,)
            Predicted probabilities in [0, 1].
        """
        return self.predict_proba(X)

    # -------------------------
    # Fit + Predict
    # -------------------------
    def fit_predict(self, X, y, **fit_kwargs):
        """
        Fit the model and predict labels for input data X.

        This is a convenience helper that forwards keyword arguments to
        `fit`. For example, you can call ``clf.fit_predict(X, y, num_epochs=10)``
        to pass training-specific options through to `fit`.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data.
        y : array-like, shape (n_samples,)
            Target labels.
        **fit_kwargs : dict
            Optional keyword arguments forwarded to `fit`.

        Returns
        -------
        labels : array-like, shape (n_samples,)
            Predicted binary labels.
        """
        self.fit(X, y, **fit_kwargs)
        return self.predict(X)

    def fit_transform(self, X, y, **fit_kwargs):
        """
        Fit the model and return transformed output for input data X.

        For this classifier the transformation is defined as `transform(X)`,
        which returns prediction probabilities/embeddings. `fit_transform`
        forwards keyword arguments to `fit`.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data.
        y : array-like, shape (n_samples,)
            Target labels.
        **fit_kwargs : dict
            Optional keyword arguments forwarded to `fit`.

        Returns
        -------
        transformed : array-like
            The result of `transform(X)` after fitting.
        """
        self.fit(X, y, **fit_kwargs)
        return self.transform(X)

    def fit_predict_on(self, X_train, y_train, X_test, **fit_kwargs):
        """
        Fit on (X_train, y_train) and predict labels for X_test.

        This convenience method forwards keyword arguments to :meth:`fit` and
        returns the result of :meth:`predict` applied to `X_test`.

        Parameters
        ----------
        X_train : array-like, shape (n_samples_train, n_features)
            Training input data.
        y_train : array-like, shape (n_samples_train,)
            Training labels.
        X_test : array-like, shape (n_samples_test, n_features)
            Test input data to predict on.
        **fit_kwargs : dict
            Optional keyword arguments forwarded to :meth:`fit`.

        Returns
        -------
        labels : array-like, shape (n_samples_test,)
            Predicted labels for `X_test`.
        """
        self.fit(X_train, y_train, **fit_kwargs)
        return self.predict(X_test)

    def fit_predict_proba_on(self, X_train, y_train, X_test, **fit_kwargs):
        """
        Fit on (X_train, y_train) and return predicted probabilities for X_test.

        This convenience method forwards keyword arguments to `fit` and
        returns the result of `predict_proba` applied to `X_test`.

        Parameters
        ----------
        X_train : array-like, shape (n_samples_train, n_features)
            Training input data.
        y_train : array-like, shape (n_samples_train,)
            Training labels.
        X_test : array-like, shape (n_samples_test, n_features)
            Test input data to predict probabilities on.
        **fit_kwargs : dict
            Optional keyword arguments forwarded to `fit`.

        Returns
        -------
        probs : array-like, shape (n_samples_test, n_classes)
            Predicted class probabilities for `X_test`.
        """
        self.fit(X_train, y_train, **fit_kwargs)
        return self.predict_proba(X_test)

    # -------------------------
    # Persistence helpers
    # -------------------------
    def save(self, path: str):
        """Save classifier minimal state to a file.

        The saved file contains:
        - VAE state_dict (on CPU)
        - feature_dim, vae_kwargs
        - classes_, threshold_, is_fitted

        This avoids pickling problematic internals.
        """
        if self.vae is None:
            raise RuntimeError(
                "No VAE instance to save. Instantiate or fit the classifier first."
            )

        # Ensure parameters are on CPU for portability
        self.vae.cpu()

        data = {
            "vae_state": self.vae.state_dict(),
            "feature_dim": self.feature_dim,
            "vae_kwargs": self.vae_kwargs,
            "classes_": getattr(self, "classes_", None),
            "threshold_": getattr(self, "threshold_", None),
            "is_fitted": getattr(self, "is_fitted", False),
        }

        torch.save(data, path)

    @classmethod
    def load(cls, path: str, device: str | None = "cpu") -> "VAEClassifier":
        """Load a classifier saved with :meth:`save`.

        Parameters
        ----------
        path : str
            Path to the saved file (torch.save output).
        device : str or torch.device, optional
            Device to place the loaded VAE on. Use "cpu" or "cuda" (if available).

        Returns
        -------
        VAEClassifier
            A classifier with instantiated `vae` and loaded parameters.
        """
        # Load to CPU first for safety, then move to device
        data = torch.load(path, map_location="cpu")

        feature_dim = data["feature_dim"]
        vae_kwargs = data.get("vae_kwargs") or {}

        clf = cls(feature_dim=feature_dim, **(vae_kwargs or {}))

        # instantiate VAE and load state
        clf.vae = VAE(clf.feature_dim, **(clf.vae_kwargs or {}))
        clf.vae.load_state_dict(data["vae_state"])

        # restore metadata
        clf.classes_ = data.get("classes_")
        clf.threshold_ = data.get("threshold_")
        clf.is_fitted = data.get("is_fitted", True)

        # move to requested device if not cpu
        if device is not None and str(device) != "cpu":
            # allow 'cuda' or torch.device
            if str(device).startswith("cuda") and torch.cuda.is_available():
                clf.vae.cuda()
            else:
                try:
                    clf.vae.to(device)
                except Exception:
                    # fallback: do nothing
                    pass

        return clf
