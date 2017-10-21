import numpy as np
from pyclustering.samples.definitions import SIMPLE_SAMPLES, FCPS_SAMPLES;

from pyclustering.cluster import cluster_visualizer;
from pyclustering.cluster.xmeans import xmeans, splitting_type;

from pyclustering.utils import read_sample, timedcall;


def template_clustering(start_centers, path, tolerance=0.025, criterion=splitting_type.BAYESIAN_INFORMATION_CRITERION,
                        ccore=False):
    sample = read_sample(path);

    xmeans_instance = xmeans(sample, start_centers, 20, tolerance, criterion, ccore);
    (ticks, result) = timedcall(xmeans_instance.process);

    clusters = xmeans_instance.get_clusters();
    print(clusters)
    print(type(clusters))
    centers = xmeans_instance.get_centers();

    criterion_string = "UNKNOWN";
    if (criterion == splitting_type.BAYESIAN_INFORMATION_CRITERION):
        criterion_string = "BAYESIAN INFORMATION CRITERION";
    elif (criterion == splitting_type.MINIMUM_NOISELESS_DESCRIPTION_LENGTH):
        criterion_string = "MINIMUM NOISELESS DESCRIPTION_LENGTH";

    print("Sample: ", path, "\nInitial centers: '", (start_centers is not None), "', Execution time: '", ticks,
          "', Number of clusters:", len(clusters), ",", criterion_string, "\n");

    visualizer = cluster_visualizer();
    visualizer.set_canvas_title(criterion_string);
    visualizer.append_clusters(clusters, sample);
    visualizer.append_cluster(centers, None, marker='*');
    visualizer.show();


def cluster_sample1_without_initial_centers():
    # cluster_path = "testSet.txt"
    template_clustering(None, cluster_path, criterion=splitting_type.BAYESIAN_INFORMATION_CRITERION);
    template_clustering(None, cluster_path,
                        criterion=splitting_type.MINIMUM_NOISELESS_DESCRIPTION_LENGTH);

def get_data(filename):
    domains = []
    file = open(filename, 'r', encoding='utf-8')
    section_numbers= 1
    for line in file.readlines():
        if line == '\n':
            section_numbers = section_numbers + 1
        else:
            line = (line.strip().split())[0]
            anchor = 'sjtu.edu.cn'
            if anchor in line:
                continue
            if line in domains:
                pass
            else:
                domains.append(line)
    domains_numbers = len(domains)
    file.close()

    # create the matrix
    matrix = np.zeros((domains_numbers, section_numbers))

    # fill the value in the matrix
    file = open('tf_idf_score.txt', 'r', encoding='utf-8')
    section_index = 0
    for line in file.readlines():
        print(line)
        anchor = 'sjtu.edu.cn'
        if line == '\n':
            section_index = section_index + 1
            continue
        line = (line.strip().split())[0]
        if anchor in line:
            continue
        else:
            print(line)
            domain_index = domains.index(line)
            matrix[domain_index, section_index] = 1
            print("matrix[", domain_index, ',', section_index, '] is 1')
    file.close()
    return domains, matrix


if __name__ == '__main__':
    domains, data_array = get_data("tf_idf_score.txt")
    matrix_file = open('matrix.txt', 'w', encoding='utf-8')
    for i in range(data_array.shape[0]):
        matrix_file.write(domains[i])
        for j in range (data_array.shape[1]):
            matrix_file.write(' ')
            matrix_file.write(str(int(data_array[i, j])))
        matrix_file.write('\n')
    matrix_file.close()

    cluster_path = "matrix.txt"
    # cluster_sample1_without_initial_centers()
