import cv2

url = "http://axis-mpr.dyndns.org/mjpg/video.mjpg"
cap = cv2.VideoCapture(url)

while True:
    ret, frame = cap.read()
    if not ret:
        print("No se pudo abrir la cámara.")
        break
    cv2.imshow("Camara IP Prueba", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
