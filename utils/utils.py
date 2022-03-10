import numpy as np
import matplotlib.pyplot as plt
import cv2
import mediapipe as mp

def diffusion_map_00(markov_chain, k=3):
    eig_values, eig_vectors = np.linalg.eig(markov_chain) #Получаем собственные вектора и собственные числа
    k_most = np.argsort(np.abs(eig_values))[-k:] #Возвращает индексы от меньшего к большему собственных чисел
    return eig_values[k_most], eig_vectors[:, k_most]#* eig_values[k_most]

def distance_vec_multi(new, base, distance_function):
    n = base.shape[0]
    dm = np.zeros([new.shape[0], n])
     
    
    for j in range(new.shape[0]):
        cur=np.expand_dims(new[j,:],0)
        diff= distance_function(np.tile(cur,[n,1]), base) 
        #print(np.tile(cur,[n,1]).shape,diff.shape)
        dm[j, :] = diff
    return dm

def plot_3d_01(slov):
    fig = plt.figure(figsize=(20, 10), dpi=80)
    ax = fig.add_subplot(projection='3d')
    
    for lbl, d in slov.items():
        ax.scatter(d[:, 0], d[:, 1], d[:, 2], label=lbl)
    ax.set_title('Diffuse map', pad=30)
    ax.view_init(azim=30)
    plt.legend()
    plt.show()

def quasydiff2D_01(a,b):
    return a-b

def gaussian_kernel_function_dff_00(epsilon, flag):
    if flag == 0:
        def kernel_function(a, b):
            return np.exp(-(np.linalg.norm( quasydiff2D_01(a,b),axis=-1) ** 2 / epsilon ** 2))
    elif flag == 2:
        def kernel_function(a, b):
            return np.linalg.norm( quasydiff2D_01(a,b))
    elif flag == 1:
        def kernel_function(a, b):
            return 1 + np.dot(a, b.T) /(np.linalg.norm(a)*np.linalg.norm(b))
            
    return kernel_function

def test(path):
    result = []
    
    mp_face_mesh = mp.solutions.face_mesh
    
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=30, refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:
        
        image = cv2.imread(path)
        output = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if output:
            for i_face, face in enumerate(output.multi_face_landmarks):
                x = []
                y = []
                z = []
                
                mp_indices = range(mp_face_mesh.FACEMESH_NUM_LANDMARKS_WITH_IRISES)
                
                for idx in mp_indices:
                    x.append(output.multi_face_landmarks[i_face].landmark[idx].x)
                    y.append(output.multi_face_landmarks[i_face].landmark[idx].y)
                    z.append(output.multi_face_landmarks[i_face].landmark[idx].z)
                result.append([x, y, z])
        
        return np.array(result)

def show(array):
    print(array.shape)
    figure = plt.figure(figsize=(10, 10))
    ax = figure.add_subplot(projection='3d')
    ax.view_init(elev=-90, azim=-90)
    
    for face in array:
        ax.scatter(face[0], face[1], face[2], c='black', s=50, alpha=1)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    
    plt.show()

def draw(path, color='#15B01A', size=10):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]

    array = test(path)
    
    figure = plt.figure(figsize=(15, 15))
    
    for face in array:
        plt.scatter(face[0]*width, face[1]*height, c=color, s=size, alpha=1)
    plt.imshow(image)
    plt.axis('off')





