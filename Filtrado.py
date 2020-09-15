
'''
/////////////////////////////////////////////
//    PONTIFICIA UNIVERSIDAD JAVERIANA     //
//                                         //
//  Carlos Daniel Cadena Cahvarro          //
//  Carlos Arturo Redondo Hurtado          //
//  Procesamiento de imagenes y vision     //
//  TALLER #2                              //
/////////////////////////////////////////////
'''

import cv2
import numpy as np
from time import time
from noise import noise
import os

if __name__ == '__main__':
    path = 'C:/Users/Carlos Cadena/Downloads'
    image_name = 'lena.png'
    path_file = os.path.join(path, image_name)
    image = cv2.imread(path_file)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_gray2 = image_gray.copy()

    # add noise
    # Ruido Gaussiano
    lena_gauss_noisy = noise("gauss", image_gray.astype(np.float) / 255)
    lena_gauss_noisy = (255 * lena_gauss_noisy).astype(np.uint8)

    image_median = cv2.medianBlur(lena_gauss_noisy, 7)  # filtrado Mediana
    image_bilateral = cv2.bilateralFilter(lena_gauss_noisy, 15, 25, 25) # filtrado Bilateral
    image_nlm = cv2.fastNlMeansDenoising(lena_gauss_noisy, 5, 15, 25) # filtrado nlm

    # Ruido s&p
    lena_sp_noisy = noise("s&p", image_gray2.astype(np.float) / 255)
    lena_sp_noisy = (255 * lena_sp_noisy).astype(np.uint8)

    image_median2 = cv2.medianBlur(lena_sp_noisy, 7) # filtrado Mediana
    image_bilateral2 = cv2.bilateralFilter(lena_sp_noisy, 15, 25, 25) # filtrado Bilateral
    image_nlm2 = cv2.fastNlMeansDenoising(lena_sp_noisy, 5, 15, 25)   # filtrado nlm

    # window size
    N = 7

    # gaussian low-pass
    image_gauss_lp = cv2.GaussianBlur(lena_gauss_noisy, (N, N), 1.5, 1.5)
    image_gauss_lp2 = cv2.GaussianBlur(lena_sp_noisy, (N, N), 1.5, 1.5)

    # Impresión de imagenes con ruido
    cv2.imshow("lena gauss noisy", lena_gauss_noisy)
    cv2.imshow("lena s&p noisy", lena_sp_noisy)

    # Impresión de imagenes filtradas (ruido Gaussiano)

    # Filtro Gaussiano LP
    cv2.imshow("Filtered Gauss LP (noise gauss)", image_gauss_lp)
    image_noise1 = abs(lena_gauss_noisy-image_gauss_lp)  # Estimación del ruido
    cv2.imshow("Estimacion de ruido", image_noise1)
    ErrorImagenFiltradaMED1 = np.sqrt((np.square(image_gray - image_gauss_lp)).mean()) # Error cuadratico medio
    print("Error cuadratico medio para filtro Gaussiano LP", ErrorImagenFiltradaMED1)
    start1 = time() # Tiempo de inicio
    image_nlm0 = cv2.fastNlMeansDenoising(image_gauss_lp, 5, 15, 25)
    ImagenFiltradaNLM1 = image_nlm0
    end1 = time()  # Tiempo de fin
    print("Tiempo de ejecución para filtro Gaussiano LP", end1 - start1, "s")

    # Filtro Mediana
    cv2.imshow("Filtered median (noise gauss)", image_median)
    image_noise2 = abs(lena_gauss_noisy - image_median) # Estimación del ruido
    cv2.imshow("Estimacion de ruido mediana", image_noise2)
    ErrorImagenFiltradaMED2 = np.sqrt((np.square(image_gray - image_median)).mean()) # Error cuadratico medio
    print("Error cuadratico medio para filtro Mediana", ErrorImagenFiltradaMED2)
    start2 = time()
    image_nlm1 = cv2.fastNlMeansDenoising(image_median, 5, 15, 25)
    ImagenFiltradaNLM2 = image_nlm1
    end2 = time()
    print("Tiempo de ejecución para filtro Mediana", end2 - start2, "s")

    # Filtro Bilateral
    cv2.imshow("Filtered bilateral (noise gauss)", image_bilateral)
    image_noise3 = abs(lena_gauss_noisy - image_bilateral) # Estimación del ruido
    cv2.imshow("Estimacion de ruido bilateral", image_noise3)
    ErrorImagenFiltradaMED3 = np.sqrt((np.square(image_gray - image_bilateral)).mean())# Error cuadratico medio
    print("Error cuadratico medio para filtro Bilateral", ErrorImagenFiltradaMED3)
    start3 = time()
    image_nlm22 = cv2.fastNlMeansDenoising(image_bilateral, 5, 15, 25)
    ImagenFiltradaNLM3 = image_nlm22
    end3 = time()
    print("Tiempo de ejecución para filtro Bilateral", end3 - start3, "s")

    # Filtro nlm
    cv2.imshow("Filtered nlm (noise gauss)", image_nlm)
    image_noise4 = abs(lena_gauss_noisy - image_nlm) # Estimación del ruido
    cv2.imshow("Estimacion de ruido nlm", image_noise4)
    ErrorImagenFiltradaMED4 = np.sqrt((np.square(image_gray - image_nlm)).mean())# Error cuadratico medio
    print("Error cuadratico medio para filtro nlm", ErrorImagenFiltradaMED4)
    start4 = time()
    image_nlm3 = cv2.fastNlMeansDenoising(image_nlm, 5, 15, 25)
    ImagenFiltradaNLM4 = image_nlm3
    end4 = time()
    print("Tiempo de ejecución para filtro nlm", end4 - start4, "s")

    # Impresión de imagenes filtradas (ruido s&p)

    # Filtro Gaussiano LP
    cv2.imshow("Filtered Gauss LP (noise S&P)", image_gauss_lp2)
    image_noise5 = abs(lena_sp_noisy - image_gauss_lp2) # Estimación del ruido
    cv2.imshow("Estimacion de ruido gauss LP S&P", image_noise5)
    ErrorImagenFiltradaMED5 = np.sqrt((np.square(image_gray - image_gauss_lp2)).mean())# Error cuadratico medio
    print("Error cuadratico medio para Gaussiano LP con ruido s&p", ErrorImagenFiltradaMED5)
    start5 = time()
    image_nlm4 = cv2.fastNlMeansDenoising(image_gauss_lp2, 5, 15, 25)
    ImagenFiltradaNLM5 = image_nlm4
    end5 = time()
    print("Tiempo de ejecución para filtro Gaussiano LP con ruido s&p", end5 - start5, "s")

    # Filtro Mediana
    cv2.imshow("Filtered median (noise S&P)", image_median2)
    image_noise6 = abs(lena_sp_noisy - image_median2) # Estimación del ruido
    cv2.imshow("Estimacion de ruido mediana S&P", image_noise6)
    ErrorImagenFiltradaMED6 = np.sqrt((np.square(image_gray - image_median2)).mean())# Error cuadratico medio
    print("Error cuadratico medio para Mediana con ruido s&p", ErrorImagenFiltradaMED6)
    start6 = time()
    image_nlm5 = cv2.fastNlMeansDenoising(image_median2, 5, 15, 25)
    ImagenFiltradaNLM6 = image_nlm5
    end6 = time()
    print("Tiempo de ejecución para filtro Mediana  con ruido s&p", end6 - start6, "s")

    # Filtro Bilateral
    cv2.imshow("Filtered bilateral (noise S&P)", image_bilateral2)
    image_noise7 = abs(lena_sp_noisy - image_bilateral2) # Estimación del ruido
    cv2.imshow("Estimacion de ruido bilateral S&P", image_noise7)
    ErrorImagenFiltradaMED7 = np.sqrt((np.square(image_gray - image_bilateral2)).mean())# Error cuadratico medio
    print("Error cuadratico medio para Bilateral con ruido s&p", ErrorImagenFiltradaMED7)
    start7 = time()
    image_nlm5 = cv2.fastNlMeansDenoising(image_bilateral2, 5, 15, 25)
    ImagenFiltradaNLM7 = image_nlm5
    end7 = time()
    print("Tiempo de ejecución para filtro Bilateral con ruido s&p", end7 - start7, "s")

    # Filtro nlm
    cv2.imshow("Filtered nlm (noise S&P)", image_nlm2)
    image_noise8 = abs(lena_sp_noisy - image_nlm2) # Estimación del ruido
    cv2.imshow("Estimacion de ruido nlm S&P", image_noise8)
    ErrorImagenFiltradaMED8 = np.sqrt((np.square(image_gray - image_nlm2)).mean())# Error cuadratico medio
    print("Error cuadratico medio para nlm con ruido s&p", ErrorImagenFiltradaMED8)
    start8 = time()
    image_nlm5 = cv2.fastNlMeansDenoising(image_nlm2, 5, 15, 25)
    ImagenFiltradaNLM8 = image_nlm5
    end8 = time()
    print("Tiempo de ejecución para filtro nlm con ruido s&p", end8 - start8, "s")

    cv2.waitKey(0)
