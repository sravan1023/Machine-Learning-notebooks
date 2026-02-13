import matplotlib.pyplot as plt


def plot_eigenfaces(pca, image_shape, explained_variance_ratio, n_eigenfaces=10):
    fig, axes = plt.subplots(2, 5, figsize=(15, 6),
                             subplot_kw={'xticks': [], 'yticks': []})
    fig.suptitle('Top 10 Eigenfaces (Principal Components)', fontsize=16, fontweight='bold')

    for i, ax in enumerate(axes.flat):
        eigenface = pca.components_[i].reshape(image_shape)
        ax.imshow(eigenface, cmap='bone')
        ax.set_title(f'PC {i+1}\n({explained_variance_ratio[i]*100:.1f}%)')

    plt.tight_layout()
    plt.savefig('eigenfaces.png', dpi=150, bbox_inches='tight')
    plt.show()
