import cv2
import numpy as np

def detect_lanes(frame):
    # Étape 1 : Convertir en niveaux de gris
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Étape 2 : Filtrage Gaussien
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Étape 3 : Détection des bords avec Canny
    edges = cv2.Canny(blur, 50, 150)

    # Étape 4 : Définir la ROI (région d’intérêt)
    height, width = edges.shape
    mask = np.zeros_like(edges)
    polygon = np.array([[
        (0, height),
        (width, height),
        (int(0.55 * width), int(0.6 * height)),
        (int(0.45 * width), int(0.6 * height))
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    cropped_edges = cv2.bitwise_and(edges, mask)

    # Étape 5 : Hough Transform pour lignes
    lines = cv2.HoughLinesP(
        cropped_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=40,
        maxLineGap=150
    )

    # Étape 6 : Dessiner les lignes sur une copie de l’image d’origine
    line_image = np.copy(frame)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 3)

    return line_image, lines
