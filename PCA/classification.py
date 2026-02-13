import time
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score


def split_data(X_scaled, y):
    return train_test_split(X_scaled, y, test_size=0.25, random_state=42, stratify=y)


def reduce_for_classification(X_train, X_test, n_components=150):
    pca_clf = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca_clf.fit_transform(X_train)
    X_test_pca = pca_clf.transform(X_test)
    return X_train_pca, X_test_pca


def train_and_evaluate(X_train, X_test, y_train, y_test, clf):
    start = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - start
    acc = accuracy_score(y_test, clf.predict(X_test))
    return acc, train_time


def run_classification(X_scaled, y, target_names, n_features, n_pca_components=150):
    X_train, X_test, y_train, y_test = split_data(X_scaled, y)
    X_train_pca, X_test_pca = reduce_for_classification(X_train, X_test, n_pca_components)

    svm_orig_acc, svm_orig_time = train_and_evaluate(
        X_train, X_test, y_train, y_test,
        SVC(kernel='rbf', class_weight='balanced', gamma='scale', random_state=42))

    svm_pca_acc, svm_pca_time = train_and_evaluate(
        X_train_pca, X_test_pca, y_train, y_test,
        SVC(kernel='rbf', class_weight='balanced', gamma='scale', random_state=42))

    knn_orig_acc, knn_orig_time = train_and_evaluate(
        X_train, X_test, y_train, y_test,
        KNeighborsClassifier(n_neighbors=5))

    knn_pca_acc, knn_pca_time = train_and_evaluate(
        X_train_pca, X_test_pca, y_train, y_test,
        KNeighborsClassifier(n_neighbors=5))

    svm_pca = SVC(kernel='rbf', class_weight='balanced', gamma='scale', random_state=42)
    svm_pca.fit(X_train_pca, y_train)
    y_pred = svm_pca.predict(X_test_pca)
    report = classification_report(y_test, y_pred, target_names=target_names)

    results = {
        'svm_orig': {'acc': svm_orig_acc, 'time': svm_orig_time},
        'svm_pca': {'acc': svm_pca_acc, 'time': svm_pca_time},
        'knn_orig': {'acc': knn_orig_acc, 'time': knn_orig_time},
        'knn_pca': {'acc': knn_pca_acc, 'time': knn_pca_time},
        'report': report,
    }
    return results


def plot_classification_results(results):
    labels = ['SVM\nOriginal', 'SVM\nPCA', 'KNN\nOriginal', 'KNN\nPCA']
    accuracies = [results['svm_orig']['acc'] * 100, results['svm_pca']['acc'] * 100,
                  results['knn_orig']['acc'] * 100, results['knn_pca']['acc'] * 100]
    times = [results['svm_orig']['time'], results['svm_pca']['time'],
             results['knn_orig']['time'], results['knn_pca']['time']]
    colors = ['#2196F3', '#4CAF50', '#2196F3', '#4CAF50']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    bars = axes[0].bar(labels, accuracies, color=colors, edgecolor='black', linewidth=0.5)
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Classification Accuracy Comparison')
    axes[0].set_ylim(0, 100)
    for bar, acc in zip(bars, accuracies):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f'{acc:.1f}%', ha='center', fontweight='bold')

    bars = axes[1].bar(labels, times, color=colors, edgecolor='black', linewidth=0.5)
    axes[1].set_ylabel('Training Time (seconds)')
    axes[1].set_title('Training Time Comparison')
    for bar, t in zip(bars, times):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(times) * 0.02,
                     f'{t:.3f}s', ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig('classification_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
