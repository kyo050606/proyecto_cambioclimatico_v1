from imageai.Detection import VideoObjectDetection
import os

# Definir la emisión promedio de CO2 por kilómetro para un automóvil estándar (en kg o L)
emission_per_km = 0.170  # Ajusta este valor según la emisión de CO2 de los autos en tu área
estimated_distance = 3.2

execution_path = os.getcwd()

def forFrame(frame_number, output_array, output_count, output_detected_frame):
    # Calcular la cantidad de CO2 emitida por cada auto detectado en este frame
    global count_of_cars
    count_of_cars = output_count['car']
    global co2_per_auto # Ajusta según sea necesario
    co2_per_auto = emission_per_km * estimated_distance 
    global co2_per_frame
    co2_per_frame = co2_per_auto * count_of_cars

    # print(f"Frame: {frame_number}, Autos detectados: {count_of_cars}, CO2 emitido: {co2_per_frame} kg")

detector = VideoObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(os.path.join(execution_path, "yolov3.pt"))
detector.loadModel()

print("Procesando video...")

video_path = detector.detectObjectsFromVideo(
    input_file_path=os.path.join(execution_path, "traffic.mp4"),
    output_file_path=os.path.join(execution_path, "traffic_detected"),
    frames_per_second=2,
    log_progress=False,
    per_frame_function=forFrame,
    return_detected_frame=True
)

print("Fin del procesamiento del video.\n")

co2_in_video = []
co2_in_video.append(co2_per_frame)
total_average_co2 = sum(co2_in_video) / len(co2_in_video)

print(f"El promedio de CO2 producido por los {count_of_cars} automoviles aparecidos en el video son {total_average_co2} kg.")

print("Fin del programa")