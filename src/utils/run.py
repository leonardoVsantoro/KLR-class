import os
import sys
script_dir = os.getcwd()
parent_dir = os.path.dirname(script_dir)
os.chdir(parent_dir)
sys.path.insert(0, parent_dir)

from src.modules import *
from src.classification import *

def getAccScores(data, classifiers, test_size = 0.2, cross_validate_KLR = False,  random_state  = 42, verbose = True, noise_scale = 0):
    """
    Evaluate classifiers on a given dataset and print accuracy scores.
    Parameters:
    data (sklearn.datasets): Dataset to evaluate.
    classifiers (dict): Dictionary of classifiers to evaluate, where keys are classifier names and values are classifier instances.
    test_size (float): Proportion of the dataset to include in the test split (default is 0.2).
    cross_validate_KLR (bool): Whether to perform cross-validation for KLRClassifier (default is False).
    random_state (int): Random seed for reproducibility (default is 42).
    verbose (bool): Whether to print detailed output during training (default is True).
    noise_scale (float): Standard deviation of Gaussian noise added to the data (default is 0).
    Returns:
    None
    """
    X, y = data.data, data.target
    if noise_scale > 0:
        rng = np.random.RandomState(random_state)
        X += rng.normal(loc=0, scale=noise_scale, size=X.shape)
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=test_size, random_state=random_state)
    if cross_validate_KLR:
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=random_state)
    else: 
        X_val, y_val = None, None

    print(f"Train size:      {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Test size:       {X_test.shape[0]} samples, {X_test.shape[1]} features")
    if X_val is not None:
        print(f"Validation size: {X_val.shape[0]} samples, {X_val.shape[1]} features")

    results = []
    for name, clf in classifiers.items():
        if verbose:
            tqdm.write(f"Training {name}...")
        start_time = time.time()
        if name == "KLRClassifier" and cross_validate_KLR :
            clf.fit(X_train, y_train, crossvalidation=cross_validate_KLR, validation_set=(X_val, y_val))
            if verbose:
                tqdm.write("   ...tuning by cross-validation")
        else:
            clf.fit(X_train, y_train)
        train_accuracy = clf.score(X_train, y_train)
        test_accuracy = clf.score(X_test, y_test)
        end_time = time.time()
        if verbose:
            tqdm.write("Complete!")

        runtime = end_time - start_time

        results.append({
            "Classifier": name,
            "Train Accuracy": train_accuracy,
            "Test Accuracy": test_accuracy,
            "Runtime (s)": runtime
        })
    results_df = pd.DataFrame(results).round(3)
    table = PrettyTable()
    table.field_names = ["Classifier", "Train Accuracy", "Test Accuracy", "Runtime (s)"]
    for _, row in results_df.iterrows():
        table.add_row(row)
    print(table)
    return None


# --------
def plot_decision_boundaries(datasets, classifiers, random_state=42, KLR_crossvalidation = False, test_size = 0.25):
    """
    Plot decision boundaries for multiple classifiers on multiple datasets.
    Parameters:
    datasets (list): List of tuples, each containing a dataset (X, y).
    classifiers (dict): Dictionary of classifiers to evaluate, where keys are classifier names and values are classifier instances.
    random_state (int): Random seed for reproducibility (default is 42).
    Returns:
    None
    """
    # cm_bright_colors = mpl_colormaps['tab10'].colors[:]
    # cm_bright = ListedColormap([cm_bright_colors[0], cm_bright_colors[3], cm_bright_colors[1], cm_bright_colors[2]])
    # cm = [plt.cm.Blues, plt.cm.Reds, plt.cm.Oranges, plt.cm.Greens, plt.cm.Purples, plt.cm.Greys, plt.cm.YlOrBr, plt.cm.YlGnBu]

    figure, axes = plt.subplots(figsize=(27, 13), ncols=len(classifiers) + 1, nrows=len(datasets))
    for i, ds in enumerate(datasets):
        X, y = ds

        n_unique_labels = len(np.unique(y))
        if n_unique_labels == 2:
            cm = [plt.cm.Reds, plt.cm.Blues]; cm_bright = ListedColormap(["#FF0000", "#0000FF"])
        else:
            cm_bright = ListedColormap(mpl_colormaps['tab10'].colors[:n_unique_labels])
            cm = [plt.cm.Blues, plt.cm.Oranges, plt.cm.Greens, plt.cm.Reds,
                plt.cm.Purples, plt.cm.Greys, plt.cm.YlOrBr, plt.cm.YlGnBu]


        X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=test_size, random_state=random_state)
        if KLR_crossvalidation:
            X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=random_state)
        else:
            X_val, y_val = None, None

        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        resolution = 75
        x_vals,y_vals = np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution)
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, resolution),
            np.linspace(y_min, y_max, resolution)
                )
        meshgrid_x, meshgrid_y = np.meshgrid(x_vals, y_vals)
        grid = np.c_[xx.ravel(), yy.ravel()]
        for j, (name, clf) in enumerate(classifiers.items()):
            j+=1
            if name == "KLRClassifier":
                clf.fit(X_train, y_train, crossvalidation=KLR_crossvalidation, validation_set = (X_val, y_val))
            else:
                clf.fit(X_train, y_train)

            score = clf.score(X_test, y_test)
            probs = clf.predict_proba(grid)
            y_pred = clf.predict(grid).reshape(meshgrid_x.shape)
            for colormap, lab in zip( cm, np.sort(np.unique(y_pred))):
                labprob = probs[:, lab].reshape(meshgrid_x.shape)
                axes[i, j].contourf(
                    meshgrid_x,
                    meshgrid_y,
                    np.where(y_pred == lab, labprob, np.nan),
                    # np.where(y_pred == lab, np.log1p(labprob), np.nan),
                    cmap=colormap,
                    alpha=0.8
                )
            axes[i,j].set_title(name)
            axes[i,j].text( x_max - 0.05*(x_max-x_min), y_min + 0.15*(y_max-y_min), ("%.2f" % score).lstrip("0"), size=15,horizontalalignment="right", )
            train_score = clf.score(X_train, y_train)
            axes[i,j].text( x_max -  0.05*(x_max-x_min), y_min + 0.05*(y_max-y_min) , ("%.2f" % train_score).lstrip("0"), size=12,horizontalalignment="right",alpha =.7)

        for j in range(0, len(classifiers) + 1):
            axes[i,j].set_xlim(x_min, x_max); axes[i,j].set_ylim(y_min, y_max)
            axes[i,j].set_xticks(()); axes[i,j].set_yticks(())
            axes[i,j].scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k")
            axes[i,j].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors="k")
        axes[i, 0].set_title("Input data")
    plt.tight_layout(); plt.show()  
    return None