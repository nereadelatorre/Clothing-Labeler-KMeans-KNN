__authors__ = 'TO_BE_FILLED'
__group__ = 'TO_BE_FILLED'

from utils_data import read_dataset, read_extended_dataset, crop_images
from Kmeans import *
from KNN import *
from utils_data import *
from utils import *
import numpy as np
import matplotlib.pyplot as plt
import time


if __name__ == '__main__':

    # Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, \
        test_color_labels = read_dataset(root_folder='./images/', gt_json='./images/gt.json')

    # List with all the existent classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))

    # Load extended ground truth
    imgs, class_labels, color_labels, upper, lower, background = read_extended_dataset()
    cropped_images = crop_images(imgs, upper, lower)
    


def Retrieval_by_color(images, color_labels, percentatges, query_colors):
    """
    Search images based on color labels and query colors.

    Args:
        images (list of numpy arrays): List of images represented as numpy arrays.
        color_labels (list of strings): List of color labels assigned to each image.
        query_colors (str or list of str): Color(s) to search for in the images.

    Returns:
        matching_images (list of numpy arrays): Images that contain the query colors in their color labels.
    """
    
    matching_images = []
    if isinstance(query_colors, str):
        query_colors = [query_colors]
    if percentatges:
        for i, labels in enumerate(color_labels):
            if all(color in labels for color in query_colors):
                for j, color in enumerate(labels):
                    if color in query_colors and percentatges[i][j] >= 0.3:
                        matching_images.append((images[i],percentatges[i][j]))
                        break
        result_images = [x[0] for x in sorted(matching_images, key=lambda x: x[1], reverse=True)]
    else:
        for i, labels in enumerate(color_labels):
            if all(color in labels for color in query_colors):
                result_images.append(images[i])
    return result_images

query_colors = ['Blue']
#matching_images = Retrieval_by_color(test_imgs, test_color_labels, None, query_colors)
#visualize_retrieval(matching_images, topN=20, info=None, ok=None, title=None, query=None)


def Retrieval_by_shape(images, class_labels, query_class):
        """
        Search images based on color labels and query colors.

        Args:
            images (list of numpy arrays): List of images represented as numpy arrays.
            color_labels (list of strings): List of color labels assigned to each image.
            query_colors (str or list of str): Color(s) to search for in the images.

        Returns:
            matching_images (list of numpy arrays): Images that contain the query colors in their color labels.
        """
        matching_images = []
        if isinstance(query_class, str):
            query_class = [query_class] 

        for i, labels in enumerate(class_labels):
            if all(clas in labels for clas in query_class):
                matching_images.append(images[i])

        return matching_images

query_classes = ['Flip Flops']
#matching_images = Retrieval_by_shape(imgs, class_labels, query_classes)
#visualize_retrieval(matching_images, topN=20, info=None, ok=None, title=None, query=None)


def Retrieval_combined(images, color_labels, class_labels, query_color, query_class, class_percentages=None, color_percentages=None):
    matching_images = []

    for i, (class_label, color_label) in enumerate(zip(class_labels, color_labels)):
        if all(clas in class_label for clas in query_class ) and all(color in color_label for color in query_color) :
            #if class_percentages is not None and color_percentages is not None:
                #if class_percentages[i] >= cl and color_percentages[i] >= co:
                #    matching_images.append(images[i])
            #else:
                matching_images.append(images[i])

    return matching_images

query_colors = ['Blue']
query_classes = ['Flip Flops']
#matching_images = Retrieval_combined(test_imgs, test_color_labels, test_class_labels, query_colors, query_classes)
#visualize_retrieval(matching_images, topN=5, info=None, ok=None, title=None, query=None)


def Kmean_statistics(images, Kmax):
    stats = {'K': [], 'WCD': [], 'Iterations': [], 'Time': []}
    iterations_data = [] #per fer boxplot
    time_data = [] # Per la nova gràfica de temps
    for k in range(2, Kmax + 1):
        i=0
        iterations_per_K = [] # per fer boxplot
        time_per_K = []
        for imatge in images:
            kmeans=KMeans(imatge, k, options = {'km_init': 'custom'})
            start_time = time.time()
            kmeans.fit()
            end_time = time.time()
            wcd = kmeans.withinClassDistance()
            iterations = kmeans.num_iter
            total_time = end_time - start_time
            stats['K'].append(k)
            stats['WCD'].append(wcd)
            stats['Iterations'].append(iterations)
            stats['Time'].append(total_time)
            iterations_per_K.append(iterations) # per fer boxplot
            time_per_K.append(total_time)
            i=i+1
        
        iterations_data.append(iterations_per_K)
        time_data.append(time_per_K) # Per la nova gràfica de temps

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.plot(stats['K'], stats['WCD'], marker='o', markersize=3, linestyle='None')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Within-Cluster Distance (WCD)')
    plt.title('Within-Cluster Distance (WCD) vs Number of Clusters')

    plt.subplot(1, 3, 2)
    plt.plot(stats['K'], stats['Time'], marker='o', markersize=3, linestyle='None')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Time (seconds)')
    plt.title('Time Taken vs Number of Clusters')

    plt.subplot(1, 3, 3)
    plt.plot(stats['K'], stats['Iterations'], marker='o', markersize=3, linestyle='None')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Iterations')
    plt.title('Iterations vs Number of Clusters')

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 6))
    plt.boxplot(iterations_data)
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Iterations')
    plt.title('Iterations vs Number of Clusters')

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(stats['K'], [sum(time_per_K) for time_per_K in time_data], marker='o', markersize=5, label='Temps real')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Total Time (seconds)')
    plt.title('Total Time vs Number of Clusters')

    plt.show()

    return stats

#stats = Kmean_statistics(test_imgs[0:100], Kmax=11)


def Visualize_Cloud(image, k):
    kmeans=KMeans(image, k)
    kmeans.fit()
    Plot3DCloud(kmeans, rows=1, cols=1, spl_id=1)
    plt.show()

#Visualize_Cloud(test_imgs[7], 5)


def Visualize_Kmeans(image, k):
    kmeans=KMeans(image, k)
    kmeans.fit()
    visualize_k_means(kmeans, image.shape)

#Visualize_Kmeans(cropped_images2[0], 5)


def Get_shape_accuracy(images, train_data, train_labels, ground_truth, k_nn):
    knn=KNN(train_data, train_labels)
    predict=knn.predict(images, k_nn)
    result=0

    for label_knn, shapes in zip(predict, ground_truth):
        if label_knn == shapes:
            result=result+1
    
    result = result/len(predict) * 100
    result = round(result, 2)
    print(result, "%" + " of class accuracy")
    return result


def Get_color_accuracy(images, ground_truth, k_max, options):
    colors = []
    for imatge in images:
        kmeans=KMeans(imatge, 1, options=options)
        kmeans._init_options(options)
        kmeans.find_bestK(k_max)
        kmeans.fit()
        labels=get_colors(kmeans.centroids)[0]
        colors.append(labels)
    
    total_correct_count=0
    total_ground_truth_count=0
    for colors_kmeans, colors_truth in zip(colors, ground_truth):
        counter=0
        count_colors=0
        repes=[]
        for color in colors_kmeans:
            if color in colors_truth:
                counter=counter+1
            if color not in repes: #evitem contar colors reptits en la classificacio
                count_colors+=1
                repes.append(color)
        if counter > count_colors: counter = count_colors
        total_correct_count+=counter
        total_ground_truth_count+=count_colors
    result = total_correct_count/total_ground_truth_count * 100
    result = round(result, 2)
    print(result, "%" + " of color accuracy")
    return result

#Get_color_accuracy(test_imgs[0:100], test_color_labels[0:100], 11, {'km_init': 'random', 'fitting': 'Fisher'})

def init_accuracy():
    init=['first', 'random', 'custom']
    colors=['blue', 'red', 'green']
    plt.figure(figsize=(12, 6))
    for i, color in zip(init, colors):
        options={'km_init': i}
        ks=[]
        acc=[]
        for k in range(2,12):
            ks.append(k)
            result=Get_color_accuracy(test_imgs[0:100], test_color_labels[0:100], k, options)
            acc.append(result)
        plt.plot(ks, acc, marker='o', markersize=4, label=i, color=color)
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Accuracy')
    plt.title('Accuracy of K-Means')
    plt.ylim(30, 75)
    plt.legend()
    plt.show()

#init_accuracy()

def gray_change_color(imatges):
    imatges=[]
    for imatge in imatges:
        im=rgb2gray(imatge)
        imatges.append(im)
    return np.array(imatges)


def millora_knn(imatges, ground_truth, train_imatges, train_class_labels):
    train_gray_list=[]
    time_data1 = []
    time_data2 = []
    ks=[]
    for imatge in train_imatges:
        im=rgb2gray(imatge)
        train_gray_list.append(im)
    train_imatges_gray=np.array(train_gray_list)
    knn_gray=KNN(train_imatges_gray, train_class_labels)

    imatges_gray_list=[]
    for imatge in imatges:
        im=rgb2gray(imatge)
        imatges_gray_list.append(im)
    imatges_gray=np.array(imatges_gray_list)
    for k in (2, 11):
        start_time = time.time()
        knn_gray.predict(imatges_gray, k)
        end_time = time.time()
        gray_time = end_time - start_time
        time_data1.append(gray_time)
        ks.append(k)

    knn=KNN(train_imatges, train_class_labels)
    for k in (2, 11):
        start_time = time.time()
        knn.predict(imatges, k)
        end_time = time.time()
        rgb_time = end_time - start_time
        time_data2.append(rgb_time)
    
    plt.figure(figsize=(12, 6))
    plt.plot(ks, time_data1, label='gray')
    plt.plot(ks, time_data2, label='color')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Time')
    plt.title('Time vs Number of Clusters with and without color')
    plt.tight_layout()
    plt.legend()
    plt.show()

    print("Temps amb les imatges en gris: ", gray_time)
    print("Temps amb les imatges en color: ", rgb_time)

    Get_shape_accuracy(imatges_gray, train_imatges_gray, train_class_labels, ground_truth, 3)
    Get_shape_accuracy(imatges, train_imatges, train_class_labels, ground_truth, 3)

#millora_knn(test_imgs, test_class_labels, train_imgs, train_class_labels)


def compare_distances_statistics(images, Kmax):
    wcd_data = []
    icd_data = []
    fisher_data = []

    for k in range(2, Kmax + 1):
        wcd_per_K = []
        icd_per_K = []
        fisher_per_K = []

        for imatge in images:
            kmeans = KMeans(imatge, k, options={'km_init': 'first'})
            kmeans.fit()
            wcd = kmeans.withinClassDistance()
            icd = kmeans.interClassDistance()
            fisher = kmeans.fisherDiscriminant()

            wcd_per_K.append(wcd)
            icd_per_K.append(icd)
            fisher_per_K.append(fisher)

        wcd_data.append(wcd_per_K)
        icd_data.append(icd_per_K)
        fisher_data.append(fisher_per_K)

    # Crear gràfics de barres per cada mètrica
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.bar(range(2, Kmax + 1), [np.mean(wcd_per_K) for wcd_per_K in wcd_data])
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Mean Within-Cluster Distance (WCD)')
    plt.title('Mean WCD vs Number of Clusters')

    plt.subplot(1, 3, 2)
    plt.bar(range(2, Kmax + 1), [np.mean(icd_per_K) for icd_per_K in icd_data])
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Mean Inter-Class Distance (ICD)')
    plt.title('Mean ICD vs Number of Clusters')

    plt.subplot(1, 3, 3)
    plt.bar(range(2, Kmax + 1), [np.mean(fisher_per_K) for fisher_per_K in fisher_data])
    plt.xlabel('Number of Clusters (K)')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Mean Fisher Discriminant Value')
    plt.title('Mean Fisher Discriminant vs Number of Clusters')

    plt.tight_layout()
    plt.show()

#compare_distances_statistics(test_imgs[0:2], 11)
