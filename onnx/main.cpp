#include <onnxruntime_cxx_api.h>

#include <vector>
#include <cstdint>
#include <iostream>

void inspect_model(Ort::Session& session){
	Ort::AllocatorWithDefaultOptions alloc;
	
	size_t num_inputs = session.GetInputCount();
	std::cout << "=== INPUTS ===" <<  std::endl;
	
	for (size_t i = 0; i < num_inputs; i++) {
		auto name = session.GetInputNameAllocated(i, alloc);
		auto info = session.GetInputTypeInfo(i);
		auto tensor_info = info.GetTensorTypeAndShapeInfo();

		std::cout << " [" << i << "] name: " << name.get() << std::endl;
		std::cout << "      type: " << tensor_info.GetElementType() << std::endl;

		std::cout << "      shape: [";
		for (auto dim : tensor_info.GetShape())
			std::cout << dim << " ";
		std::cout << "]" << std::endl;
	}
	size_t num_outputs = session.GetOutputCount();
	for (size_t i = 0; i < num_inputs; i++) {
		auto name = session.GetOutputNameAllocated(i, alloc);
		auto info = session.GetOutputTypeInfo(i);
		auto tensor_info = info.GetTensorTypeAndShapeInfo();

		std::cout << " [" << i << "] name: " << name.get() << std::endl;
		std::cout << "      type: " << tensor_info.GetElementType() << std::endl;

		std::cout << "      shape: [";
		for (auto dim : tensor_info.GetShape())
			std::cout << dim << " ";
		std::cout << "]" << std::endl;
	}
}


int main(){
	//Esto se hace una vez por programa
	Ort::Env env (ORT_LOGGING_LEVEL_WARNING, "MyApp");
	//Opciones de la session, hay 3, la de threads, la de exceutaion, y la de graph optimizations
	Ort::SessionOptions session_options;
	//Aqui se inicia la session con el archivo .onnx
	Ort::Session session(env, "../mnist.onnx", session_options);
	//Esto se hace una vez por programa tambien, porque es expensive
	
	//4 pasos para crear el tensor
	//descibir la shape, 1 batch, 28 filas, con 28 columnas
	std::vector<int64_t> input_shape = {1, 28, 28};
	
	//Prepara la data
	std::vector<float> input_data(784, 0.0f);  //aqui estan todos en 0
	
	//Allocar ino
	Ort::MemoryInfo mem_info = 
		Ort::MemoryInfo::CreateCpu(
			OrtArenaAllocator, OrtMemTypeDefault);
	
	//Una vez hecho eso, se crea el tensor
	Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
	mem_info,
	input_data.data(),
	input_data.size(),
	input_shape.data(),
	input_shape.size()
	);
	
	//Session 4.1
	Ort::AllocatorWithDefaultOptions allocator;
	
	//Para agarrar el nombre 0
	auto input_name_ptr = session.GetInputNameAllocated(0, allocator);
	std::string input_name = input_name_ptr.get();

	//Para agarrar el output 0
	auto output_name_ptr = session.GetOutputNameAllocated(0, allocator);
	std::string output_name = output_name_ptr.get();
	
	std::cout << "Input: " << input_name << std::endl;
	std::cout << "Output: " << output_name << std::endl;
	//Esto es para saber los NOMBRES, uno se llama input, el otro Identity:0

	//Ahora vamos a hacer un Run Call
	const char* input_names[] = { input_name.c_str() };
	const char* output_names[] = { output_name.c_str() };
	
	auto output_tensors = session.Run(
	Ort::RunOptions(nullptr),
	input_names,
	&input_tensor,
	1,
	output_names,
	1
	);

	//Para ya ver la prediccion
	float* output_data = output_tensors[0].GetTensorMutableData<float>();
	int num_classes = 10;
	int predicted_class = 0;
	float max_score = output_data[0];
	
	for (int i = 1; i < num_classes; i++){
	if(output_data[i] > max_score) {
		max_score = output_data[i];
		predicted_class = i;
	}
	}
	std::cout << "Predicted digit: " << predicted_class << std::endl;
	
	inspect_model(session);
	return 0;
}

