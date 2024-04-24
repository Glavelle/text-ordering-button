import streamlit as st
from hilbertcurve.hilbertcurve import HilbertCurve
import zCurve as z

from sentence_transformers import SentenceTransformer
import umap.umap_ as mp
import umap.plot

from scipy.spatial.distance import pdist, squareform

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

from numpy.linalg import norm
from scipy.spatial.transform import Rotation as R

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

from transformers import AutoTokenizer

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'



# Example usage with your sort_by_z_order function
def sort_by_z_order_2(df, points_column='Points', bits_per_dim=10):
    # First, scale the data points
    df = scale_data_points(df, points_column, bits_per_dim)

    points_int = df['Scaled Points'].tolist()
    # Assuming 'z.par_interlace' is defined and works as expected
    morton_codes = z.par_interlace(points_int, dims=3, bits_per_dim=bits_per_dim)

    # Pair each Morton code with its index in the original DataFrame
    indexed_morton_codes = list(zip(morton_codes, range(len(morton_codes))))

    # Sort the indexed Morton codes
    indexed_morton_codes.sort()

    # Extract the sorted indices
    sorted_indices = [index for _, index in indexed_morton_codes]

    # Use the sorted indices to reorder the original DataFrame
    sorted_df = df.iloc[sorted_indices].reset_index(drop=True)


    start= sorted_df['Points'].iloc[0]

    #print(f'Start point z order:{start}')
    #display(sorted_df)

    return sorted_df

def perform_pca(Lab_values):
    # Standardize the data
    scaler = StandardScaler()
    color_points_scaled = scaler.fit_transform(Lab_values)

    # Apply PCA
    pca = PCA(n_components=1)
    principal_component = pca.fit_transform(color_points_scaled)

    # The principal axis (direction) in the original feature space
    principal_axis = pca.components_.flatten()

    mean_lab = np.mean(Lab_values, axis=0)

    return principal_component, principal_axis, mean_lab


def rotate_data(Lab_values, principal_axis, mean_lab):
    # Normalise the principal component and target vector
    pc1 = principal_axis
    target_vector = np.array([1, 1, 1]) / norm([1, 1, 1])

    # Calculate rotation axis and angle
    rotation_axis = np.cross(pc1, target_vector) / norm(np.cross(pc1, target_vector))
    angle = np.arccos(np.dot(pc1, target_vector))

    # Rotate the dataset
    rotation_matrix = R.from_rotvec(rotation_axis * angle).as_matrix()
    Lab_values_centered = Lab_values - mean_lab
    Lab_values_rotated = np.dot(Lab_values_centered, rotation_matrix)

    pc1_rotated = np.dot(rotation_matrix, pc1)

    mean_lab_rotated = np.mean(Lab_values_rotated, axis=0)

    return Lab_values_rotated, pc1_rotated, mean_lab_rotated

def scale_data_points(df, points_column='Points', bits_per_dim=10):

    # Determine the target maximum value based on bits_per_dim
    max_value_dim = 2**bits_per_dim - 1

    # Convert DataFrame column of points to array
    points_array = np.array(df[points_column].tolist())

    # Calculate the min and max across all dimensions (columns) of the points
    min_data_dim = points_array.min(axis=0)
    max_data_dim = points_array.max(axis=0)

    # Scale points to the range [0, max_value_dim]
    scaled_points = (points_array - min_data_dim) / (max_data_dim - min_data_dim) * max_value_dim

    # Round and convert to integers since Morton codes operate on integers
    scaled_points_int = np.round(scaled_points).astype(int)

    # Convert scaled points back to a list of lists
    scaled_points_list = scaled_points_int.tolist()

    df['Scaled Points'] = scaled_points_list

    return df

def encode_sentences_and_reduce_to_3d(df,sent_column='Sentences', nn=15):
    """
    Encodes a list of sentences to high-dimensional vectors using a Sentence-BERT model,
    then reduces the dimensionality to 3D using UMAP.

    """
    # Load the Sentence-BERT model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    sentences=df[sent_column].tolist()

    # Encode the sentences
    sentence_embeddings = model.encode(sentences, show_progress_bar=False)

    # Initialize UMAP
    reducer = mp.UMAP(n_neighbors=nn, n_components=3, metric='cosine')

    # Reduce the dimensionality of the sentence embeddings to 3D
    embedding_3d = reducer.fit_transform(sentence_embeddings)

    df['Points']=list(embedding_3d)
    df['Points 384'] = list(sentence_embeddings)

    return df

def ordering_algorithm(Sentences):

    sent_df =pd.DataFrame({
        'Sentences': Sentences
    })

    df_titles = encode_sentences_and_reduce_to_3d(sent_df,sent_column='Sentences')

    Lab_points = df_titles['Points'].tolist()

    # rotate
    principal_component, pc1, mean_lab = perform_pca(Lab_points)

    #Rotate Data to angle [0.57,0.57,0.57]]
    Lab_values_rotated, pc1_rotated, mean_lab_rotated = rotate_data(Lab_points, pc1, mean_lab)
    df_titles['Points Rotated'] = Lab_values_rotated.tolist()
    concat_df =scale_data_points(df_titles, points_column='Points Rotated', bits_per_dim=10)


    #sort points and visualise
    z_order_sort = sort_by_z_order_2(concat_df, points_column='Scaled Points', bits_per_dim=10)
    #hilbert_sort = sort_by_hilbert_curve_2(concat_df, points_column='Scaled Points', order=10)

    return z_order_sort

def main():
    st.title('Sentence Sorter - Sort Sentences by their Meaning')
    st.markdown('Please insert a .txt file where each sentence or line of text is seperated by a newline character.')

    uploaded_file = st.file_uploader("Choose a file", type=['txt'])
    if uploaded_file is not None:
        st.write('Ordering...')
        content = uploaded_file.getvalue().decode("utf-8")
        sentences = content.split('\n')
        result_df = ordering_algorithm(sentences)
        ordered_sentences = result_df['Sentences'].tolist()
        st.write("Ordered Sentences:")
        for sentence in ordered_sentences:
            st.write(sentence)


if __name__ == "__main__":
    main()