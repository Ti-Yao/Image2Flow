from scipy.sparse import coo_matrix, csr_matrix
import pymeshlab as ml
import numpy as np
import stl 
from skimage import measure
import trimesh
import networkx
import matplotlib.pyplot as plt
from scipy.ndimage import label, generate_binary_structure, binary_erosion, binary_dilation, center_of_mass
import tensorflow as tf
import pyvista as pv
import sys
from scipy.spatial.distance import directed_hausdorff, cdist
import pandas as pd
from spektral.utils import chebyshev_filter
import pymeshfix as mf
from tqdm import tqdm
from tensorflow_addons.layers import InstanceNormalization
from spektral.layers import ChebConv
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Conv3D, SpatialDropout3D, UpSampling3D, MaxPooling3D, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Concatenate, Average, LeakyReLU, BatchNormalization
import tensorflow.keras.backend as K
import shapely
from rasterio import features
import rasterio
from shapely.geometry import Polygon
from tensorflow.keras.layers import Layer, Add, Activation
from tensorflow.keras.models import Model

sys.path.append('/home/tina/Project/Force-ML/training')
sys.path.append('/home/tina/Project/Pipeline')

BATCH_SIZE = 1


def calculate_normals(vertices, faces):
    # Extract the vertex coordinates for each face
    v0 = tf.gather(vertices, faces[:, 0])
    v1 = tf.gather(vertices, faces[:, 1])
    v2 = tf.gather(vertices, faces[:, 2])
    edge1 = v1 - v0
    edge2 = v2 - v0
    # Compute the face normals using the cross product
    face_normals = tf.linalg.cross(edge1, edge2)

    # Normalize the face normals
    face_normals = tf.math.l2_normalize(face_normals, axis=1)
    return face_normals

def find_overlapping_indices(N, M, tolerance=1e-5):
    distance_matrix = np.linalg.norm(N[:, np.newaxis] - M, axis=2)
    overlapping_mask = np.any(distance_matrix <= tolerance, axis=1)
    overlapping_indices = np.where(overlapping_mask)[0]
    return overlapping_indices

def extract_cap_faces(surface_template_mesh, cell_id):
    cell_entity_ids_to_extract = np.where(surface_template_mesh['CellEntityIds'] == cell_id)[0]
    cap_mesh = surface_template_mesh.extract_cells(cell_entity_ids_to_extract)
    cap_indices_a = cap_mesh['vtkOriginalPointIds']
    cap_mesh = cap_mesh.extract_surface().triangulate()
    cap_faces = cap_mesh.faces.reshape(-1, 4)[:, 1:]
    cap_edges = trimesh.Trimesh(cap_mesh.points, cap_faces).edges
    cap_points = cap_mesh.points
    cap_indices_b = cap_mesh['vtkOriginalPointIds']
    return [cap_indices_a, cap_indices_b], cap_faces, cap_edges

def calculate_assd(surface1_points, surface2_points):
    # Calculate distances between each point on surface1 and surface2
    distances_surface1_to_surface2 = cdist(surface1_points, surface2_points)
    distances_surface2_to_surface1 = cdist(surface2_points, surface1_points)

    # Calculate ASSD by averaging the distances in both directions
    assd = (np.mean(np.min(distances_surface1_to_surface2, axis=1)) + np.mean(np.min(distances_surface2_to_surface1, axis=1))) / 2

    return assd

def extract_wall_faces(clipped_mesh, cell_id):
    wall_mesh = pv.read(f'wall_{cell_id}.vtk').extract_surface().triangulate()
    overlap = find_overlapping_indices(clipped_mesh.points, wall_mesh.points) 
    wall_mesh = clipped_mesh.extract_points(overlap)
    wall_indices_a = wall_mesh['vtkOriginalPointIds']
    wall_mesh = wall_mesh.extract_surface().triangulate()
    wall_faces = wall_mesh.faces.reshape(-1, 4)[:, 1:]
    wall_points = wall_mesh.points
    wall_indices_b = wall_mesh['vtkOriginalPointIds']
    return [wall_indices_a, wall_indices_b], wall_faces


def orthogonal_loss(pred_coords, cap_indices, cap_faces, wall_indices, wall_faces):
    pred_coords = pred_coords[0]
    #pred_edges = tf.gather(pred_coords, surface_vertex_indices)

    cap_coords = tf.gather(pred_coords, cap_indices[0])
    cap_coords = tf.gather(cap_coords, cap_indices[1])
    cap_normals = calculate_normals(cap_coords, cap_faces)

    wall_coords = tf.gather(pred_coords, clipped_indices[0])
    wall_coords = tf.gather(wall_coords, clipped_indices[1])
    wall_coords = tf.gather(wall_coords, wall_indices[0])
    wall_coords = tf.gather(wall_coords, wall_indices[1])
    wall_normals = calculate_normals(wall_coords, wall_faces)

    dot_products = tf.abs(tf.matmul(wall_normals, tf.transpose(cap_normals)))

    orthogonal_loss = tf.reduce_sum(dot_products)

    return orthogonal_loss

def get_template_aspect_ratio(template_mesh):
    template_edges = tf.gather(template_mesh.points, template_mesh.cells.reshape(-1, 5)[:, 1:])
    diff = template_edges[:, :-1] - template_edges[:, 1:]
    lengths = tf.norm(diff, axis=2)
    aspect_ratios = tf.reduce_max(lengths, axis=1) / tf.reduce_min(lengths, axis=1)
    template_aspect_ratio = tf.reduce_mean(aspect_ratios)
    return template_aspect_ratio.numpy()

def get_template_dev_edge(template_mesh, surface):
    if surface:
        template_mesh = template_mesh.extract_surface().triangulate()
        template_edges_lines = tf.gather(template_mesh.points, template_mesh.faces.reshape(-1, 4)[:, 1:])
    else:
        template_edges_lines = tf.gather(template_mesh.points, template_mesh.cells.reshape(-1, 5)[:, 1:])
    template_edge_lengths = calculate_edge_length(template_edges_lines)
    template_mean_edge_length = tf.reduce_mean(template_edge_lengths)
    template_std_edge_length = tf.math.reduce_std(template_edge_lengths)
    dev_edge_length = tf.divide(template_std_edge_length, template_mean_edge_length) 
    dev_edge_length = tf.maximum(dev_edge_length, 0)
    return dev_edge_length.numpy()
    
def hausdorff_distance(points1, points2):
    # Compute directed Hausdorff distances
    d1 = directed_hausdorff(points1, points2)[0]
    d2 = directed_hausdorff(points2, points1)[0]

    return max(d1, d2)

class Projection(Layer):
    def __init__(self, **kwargs):
        super(Projection, self).__init__(**kwargs)

    def call(self, inputs):
        image_features = inputs[0]
        graph_features = inputs[1]
        image_height = tf.shape(image_features)[1]
        x, y, z  = graph_features[...,-3], graph_features[...,-2], graph_features[...,-1]
        factor = tf.cast(128 / image_height, tf.float32)
                
        x = x / factor
        y = y / factor
        z = z / factor

        x1 = tf.floor(x)
        x2 = tf.minimum(tf.math.ceil(x), tf.cast(image_height - 1, tf.float32))
        y1 = tf.floor(y)
        y2 = tf.minimum(tf.math.ceil(y), tf.cast(image_height - 1, tf.float32))
        z1 = tf.floor(z)
        z2 = tf.minimum(tf.math.ceil(z), tf.cast(image_height - 1, tf.float32))

        q11 = tf.gather_nd(image_features[0], tf.cast(tf.stack([x1, y1, z1], axis=-1), tf.int32))
        q21 = tf.gather_nd(image_features[0], tf.cast(tf.stack([x2, y1, z1], axis=-1), tf.int32))
        q12 = tf.gather_nd(image_features[0], tf.cast(tf.stack([x1, y2, z1], axis=-1), tf.int32))
        q22 = tf.gather_nd(image_features[0], tf.cast(tf.stack([x2, y2, z1], axis=-1), tf.int32))

        wx = tf.expand_dims(tf.subtract(x, x1), -1)
        wx2 = tf.expand_dims(tf.subtract(x2, x), -1)
        lerp_x1 = tf.add(tf.multiply(q21, wx), tf.multiply(q11, wx2))
        lerp_x2 = tf.add(tf.multiply(q22, wx), tf.multiply(q12, wx2))
        wy = tf.expand_dims(tf.subtract(y, y1), -1)
        wy2 = tf.expand_dims(tf.subtract(y2, y), -1)
        lerp_y1 = tf.add(tf.multiply(lerp_x2, wy), tf.multiply(lerp_x1, wy2))

        q11 = tf.gather_nd(image_features[0], tf.cast(tf.stack([x1, y1, z2], axis=-1), tf.int32))
        q21 = tf.gather_nd(image_features[0], tf.cast(tf.stack([x2, y1, z2], axis=-1), tf.int32))
        q12 = tf.gather_nd(image_features[0], tf.cast(tf.stack([x1, y2, z2], axis=-1), tf.int32))
        q22 = tf.gather_nd(image_features[0], tf.cast(tf.stack([x2, y2, z2], axis=-1), tf.int32))
        lerp_x1 = tf.add(tf.multiply(q21, wx), tf.multiply(q11, wx2))
        lerp_x2 = tf.add(tf.multiply(q22, wx), tf.multiply(q12, wx2))
        lerp_y2 = tf.add(tf.multiply(lerp_x2, wy), tf.multiply(lerp_x1, wy2))

        wz = tf.expand_dims(tf.subtract(z, z1), -1)
        wz2 = tf.expand_dims(tf.subtract(z2, z),-1)
        lerp_z = tf.add(tf.multiply(lerp_y2, wz), tf.multiply(lerp_y1, wz2))
        return lerp_z







def sparse_csr_to_tf(csr_mat):
    indptr = tf.constant(csr_mat.indptr, dtype=tf.int64)
    elems_per_row = indptr[1:] - indptr[:-1]
    i = tf.repeat(tf.range(csr_mat.shape[0], dtype=tf.int64), elems_per_row)
    j = tf.constant(csr_mat.indices, dtype=tf.int64)
    indices = np.stack([i, j], axis=-1)
    data = tf.constant(csr_mat.data)
    return tf.sparse.SparseTensor(indices, data, csr_mat.shape)


def is_closed(poly):
    closed = False
    try:
        poly.interiors[0]
        closed = True
    except:
        pass
    return closed

def mask_to_polygons_layer(mask):
    all_polygons = []
    for shape, value in features.shapes(mask.astype(np.int16), mask=(mask >0), transform=rasterio.Affine(1.0, 0, 0, 0, 1.0, 0)):
        all_polygons.append(shapely.geometry.shape(shape))

    all_polygons = shapely.geometry.MultiPolygon(all_polygons)
    if not all_polygons.is_valid:
        all_polygons = all_polygons.buffer(0)
        # Sometimes buffer() converts a simple Multipolygon to just a Polygon,
        # need to keep it a Multi throughout
        if all_polygons.geom_type == 'Polygon':
            all_polygons = shapely.geometry.MultiPolygon([all_polygons])
            
    return all_polygons





def find_min_bounding_cube(volume, crop_factor = 1.2):
    # Find the indices of occupied voxels along each axis
    x_indices, y_indices, z_indices = np.where(volume == 1)

    # Find the minimum and maximum values along each axis
    x_min = np.min(x_indices)
    x_max = np.max(x_indices)
    y_min = np.min(y_indices)
    y_max = np.max(y_indices)
    z_min = np.min(z_indices)
    z_max = np.max(z_indices)

    # Calculate the current sizes along each dimension
    x_size = x_max - x_min 
    y_size = y_max - y_min
    z_size = z_max - z_min 

    # Find the maximum size among the three dimensions
    max_size = max(x_size, y_size, z_size) * crop_factor

    # Calculate the target size that is both equal and a factor of 32
    largest_side = int((np.ceil(max_size + 8) // 8) * 8)
    
    x_mid = round((x_max + x_min)/2)
    y_mid = round((y_max + y_min)/2)
    z_mid = round((z_max + z_min)/2)
    
    half_largest_side = round(largest_side/2)
    x_max, x_min = x_mid + half_largest_side, x_mid - half_largest_side
    y_max, y_min = y_mid + half_largest_side, y_mid - half_largest_side
    z_max, z_min = z_mid + half_largest_side, z_mid - half_largest_side
    
    if x_min < 0:
        x_min = 0
        x_max -= x_min
    if y_min < 0:
        y_min = 0
        y_max -= y_min
    if z_min < 0:
        z_min = 0
        z_max -= z_min
    # Return the coordinates of the minimum bounding cube
    return x_min, x_max, y_min, y_max, z_min, z_max

def roll_image_to_center(image, center):
    center = [int(round(coord)) for coord in center]
    # Calculate the amount of rolling for each axis
    roll_x = image.shape[0] // 2 - center[0]
    roll_y = image.shape[1] // 2 - center[1]
    roll_z = image.shape[2] // 2 - center[2]

    # Roll the image
    rolled_image = np.roll(image, roll_x, axis=0)
    rolled_image = np.roll(rolled_image, roll_y, axis=1)
    rolled_image = np.roll(rolled_image, roll_z, axis=2)

    # Set the rolled border pixels to zero
    if roll_x > 0:
        rolled_image[:roll_x, :, :] = 0
    elif roll_x < 0:
        rolled_image[roll_x:, :, :] = 0

    if roll_y > 0:
        rolled_image[:, :roll_y, :] = 0
    elif roll_y < 0:
        rolled_image[:, roll_y:, :] = 0

    if roll_z > 0:
        rolled_image[:, :, :roll_z] = 0
    elif roll_z < 0:
        rolled_image[:, :, roll_z:] = 0

    return rolled_image


def resize_3d(image, size= 128):
    image = np.moveaxis(image, 0,-1)
    image = np.array(tf.image.resize(image[...,np.newaxis], (size, size))[...,0])
    image = np.moveaxis(image, -1,0)
    image = np.array(tf.image.resize(image[...,np.newaxis], (size, size))[...,0])
    return image

def vtu_adjacency(mesh):
    
    # Get the cell connectivity
    edges = mesh.cells.reshape(-1, 5)[:, 1:]
    # Compute the number of nodes
    num_nodes = np.max(edges) + 1
    # Create an empty adjacency matrix
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    # Populate the adjacency matrix
    for cell in edges:
        for i in range(4):
            for j in range(i+1, 4):
                adj_matrix[cell[i], cell[j]] = 1
                adj_matrix[cell[j], cell[i]] = 1
                
    return adj_matrix


def calculate_edge_length(edges_lines):
    all_edge_lengths = []
    for j in range(edges_lines.shape[1] - 1):
        edges_lines_start = edges_lines[:, j, :]
        edges_lines_end = edges_lines[:, j + 1, :]
        edge_lengths = tf.norm(edges_lines_start - edges_lines_end, axis = -1) 
        all_edge_lengths.append(edge_lengths)
    all_edge_lengths = tf.stack(all_edge_lengths, axis=-1)
    return all_edge_lengths

def mean_edge_length(pred_coords):
    pred_coords = pred_coords[0]
    pred_edges_lines = tf.gather(pred_coords, template_edges)
    pred_edge_lengths = calculate_edge_length(pred_edges_lines)
    pred_mean_edge_length = tf.reduce_mean(pred_edge_lengths)
    
    template_mean_edge_length = tf.reduce_mean(template_edge_lengths)
    
    mean_diff_edge_length = tf.abs(template_mean_edge_length - pred_mean_edge_length) 
    return mean_diff_edge_length


def aspect_ratio_length(pred_coords):
    pred_coords = pred_coords[0]
    pred_edges_lines = tf.gather(pred_coords, template_edges)
    pred_edge_lengths = calculate_edge_length(pred_edges_lines)
    
    pred_max_edge_length = tf.reduce_max(pred_edge_lengths)
    pred_min_edge_length = tf.reduce_min(pred_edge_lengths)
    aspect_ratio = tf.divide(pred_max_edge_length, pred_min_edge_length) - 1
    return aspect_ratio



def dice(y_true, y_pred, const=K.epsilon()):
    # tf tensor casting
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    
    # <--- squeeze-out length-1 dimensions.
    y_pred = tf.squeeze(y_pred)
    y_true = tf.squeeze(y_true)
    
    loss_val = 1 - dice_coef(y_true, y_pred, const=const)
    
    return loss_val

def dice_coef(y_true, y_pred, const=K.epsilon()):
    # flatten 2-d tensors
    y_true_pos = tf.reshape(y_true, [-1])
    y_pred_pos = tf.reshape(y_pred, [-1])
    
    # get true pos (TP), false neg (FN), false pos (FP).
    true_pos  = tf.reduce_sum(y_true_pos * y_pred_pos)
    false_neg = tf.reduce_sum(y_true_pos * (1-y_pred_pos))
    false_pos = tf.reduce_sum((1-y_true_pos) * y_pred_pos)
    
    # 2TP/(2TP+FP+FN) == 2TP/()
    coef_val = (2.0 * true_pos + const)/(2.0 * true_pos + false_pos + false_neg)
    
    return coef_val
    
def single_dice(im1, im2):
    im1 = np.asarray(im1).astype(bool)
    im2 = np.asarray(im2).astype(bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / (im1.sum() + im2.sum())

    return largest_component_volume
