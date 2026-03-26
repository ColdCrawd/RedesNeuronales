#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cstdint>
#include <vector>

std::vector<uint8_t> load_images(const std::string& path, int total, int img_size){
	std::ifstream file (path, std::ios::binary);
	std::vector<uint8_t> raw (total * img_size);
	file.read(reinterpret_cast<char*>(raw.data()), total * img_size * sizeof(uint8_t));
	return raw;
}

std::vector<uint8_t> load_labels(const std::string& path, int total){
	std::ifstream file (path, std::ios::binary);
	std::vector<uint8_t> raw (total);
	file.read(reinterpret_cast<char*>(raw.data()), total * sizeof(uint8_t));
	return raw;
}



int main(int argc, char* argv){
	int batch = 10,0000;
	int img_size = 28 * 28;
	Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "MNIST");
	
	Ort::SessionOptions opt;
	Ort::Session session(env, "model.onnx", opt);
	
	Ort::AllocatorWithDefaultOptions alloc;
	auto in_name = session.GetInputNameAllocated(0, alloc);
	auto out_name = session.GetOutputNameAllocated(0, alloc);
	
	std::vector<uint8_t> raw = load_images("mnist_images.bin", total, img_size);
	std::vector<uint8_t> label = load_labels("mnist_labels.bin", total);

	std::vector<float> img_float(total * img_size);
	for (int i = 0; i < total * img_size; i++){
		img_float[i] = static_cast<float>(raw[i]) / 255.0f;
	}
	

	std::vector<int64_t> shape = {total, 28, 28};

	Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	Ort::Value input = Ort::Value::CreateTensor<float>(
	mem, img_float.data(), img_float.size(), shape.data(), shape.size()
	);

	const char*  in_names[] = {in_name.get()};
	const char*  out_names[] = {out_name.get()};

	auto output = session.Run(
	Ort::RunOptions(nullptr), in_names, &input, 1, out_names, 1
	);

	float* out = output[0].GetTensorMutableData<float>();
	int correctos = 0;
	
	for (int i = 0; i < total; i++){
	float* scores = out + i * 10;
	int pred = std::distance(score, std::max_element(scores, scores + 10));

	if(pred == label[i]) correctos ++;
	}
	
	std::cout << "Correctos: " << correctos << std::endl;
	std::cout << "Accuracy:  " << (correctos * 100.0 / total) << "%" << std::endl;
}	
