#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include <iosfwd>
#include <iostream>
#include <string>
#include <sstream>
#include <ctime>

// bitmap decoder
class BmpDecoder{
private:
	int width;
	int height;
public:
	BmpDecoder(int width, int height): width(width), height(height){

	}

	// decodes
	void SaveAsFile(unsigned char* data, const char* filename){

		FILE *f;

		// 54 bytes header + number of pixels * number of channels
		int filesize = 14 + 40 + 3*width*height;

		// 14 bytes for header
		unsigned char bmpfileheader[14] = {'B','M', 0,0,0,0, 0,0, 0,0, 54,0,0,0};
		// 40 bytes (version from Windows 3.1x - without alpha channel, color space and ICC profiles)
		unsigned char bmpinfoheader[40] = {0,0,0,0, 0,0,0,0, 0,0,0,0, 1,0, 24,0};

		// size of each row is rounded up to a multiple of 4B by padding
		unsigned char bmppad[3] = {0,0,0};

		// file size, 4B
		bmpfileheader[ 2] = (unsigned char)(filesize    );
		bmpfileheader[ 3] = (unsigned char)(filesize>> 8);
		bmpfileheader[ 4] = (unsigned char)(filesize>>16);
		bmpfileheader[ 5] = (unsigned char)(filesize>>24);

		// header size, 1B
		bmpinfoheader[ 0] = (unsigned char)(          40);
		// width in pixels, 4B
		bmpinfoheader[ 4] = (unsigned char)(       width);
		bmpinfoheader[ 5] = (unsigned char)(   width>> 8);
		bmpinfoheader[ 6] = (unsigned char)(   width>>16);
		bmpinfoheader[ 7] = (unsigned char)(   width>>24);
		// height in pixels, 4B
		bmpinfoheader[ 8] = (unsigned char)(  height    );
		bmpinfoheader[ 9] = (unsigned char)(  height>> 8);
		bmpinfoheader[10] = (unsigned char)(  height>>16);
		bmpinfoheader[11] = (unsigned char)(  height>>24);

		// write data into bmp file
		f = fopen(filename,"wb");
		fwrite(bmpfileheader,1,14,f);
		fwrite(bmpinfoheader,1,40,f);
		for(int i=0; i<height; i++)
		{
			// write one row and pad to have size of each row a multiple of 4B
			fwrite(data+(width*(height-i-1)*3),3,width,f);
			fwrite(bmppad,1,(4-(width*3)%4)%4,f);
		}
		fclose(f);
	}
};

// error logger
static void CudaSafe(cudaError_t err,
					 const char *file,
					 int line ) {
						 if (err != cudaSuccess) {
							 printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
								 file, line );
							 exit( EXIT_FAILURE );
						 }
}

// simplified error macro
#define CUDA_SAFE( err ) (CudaSafe( err, __FILE__, __LINE__ ))

// structure that is passed to the GPU memory
// holds computation parameters
struct Perlinfo{
	int width;
	int height;
	float persistence;
	int octaves;
	float zoom;
	int seed;
};


cudaError_t CUDAPerlinNoise(Perlinfo info, unsigned char* outputBitmap, int blocks, int threads, bool useBuffer);
cudaError_t CUDAWriteInfo(Perlinfo info);
void WriteResult(Perlinfo info, int threads, int blocks, bool buffer);

// linear coherent-noise function
__device__ float Noise(int x, int y, int seed)
{
	int n = x+y*seed;
	// spread information contained in input value across all bits
	n ^= (n << 13);
	// spread information across all domains, using three magnitudes
	n = (n * (n * n * (15731) + (789221)) + (1376312589));
	// scale it down to the interval 0..2 and correct to -1..1
	float output = (float)(1.0 - (n & 0x7fffffff) / 1073741824.0);
	return output;
}


// cosine interpolation
// v1: first value to interpolate between
// v2: second value to interpolate between
// x: interpolating parameter
__device__ float CosineInterpolation(float v1, float v2, float x)
{
	float f = (float)(1 - cos(x * 3.1415927)) * 0.5f;
	return v1 * (1 - f) + v2 * f;
}


// ============================================= VARIATION 1:: COMPUTE AND WRITE RESULT IMMEDIATELY ===============================================

// calculates simple 3x3 smooth
__device__	float SmoothNoise(int x, int y, int seed)
{
	/*
	*  |x-1,y+1| x ,y+1|x+1,y+1|
	*  |x-1,y  | x ,y  |x+1,y  |
	*  |x-1,y-1| x ,y-1|x+1,y-1|
	*/

	// smooth corners (left upper, left down etc.)
	float corners = (Noise(x - 1, y - 1, seed) + Noise(x + 1, y - 1, seed) + Noise(x - 1, y + 1, seed) + Noise(x + 1, y + 1, seed)) / 16;
	// smooth sides (left, right,...)
	float sides = (Noise(x - 1, y, seed) + Noise(x + 1, y, seed) + Noise(x, y - 1, seed) + Noise(x, y + 1, seed)) / 8;
	// smooth center
	float center = Noise(x, y, seed) / 4;
	// sum values
	return corners + sides + center;
}

// calculates interpolated noise, using cosine transform and 3x3 smooth algorithm
__device__ float Interpolate(float x, float y, int seed){

float i1, i2;
// interpolate according to X for bottom part, parameter is fraction part of x
i1 = CosineInterpolation(SmoothNoise((int)x, (int)y, seed), SmoothNoise((int)(x + 1), (int)(y), seed), x - ((int)(x)));
// interpolate according to X for upper part
i2 = CosineInterpolation(SmoothNoise((int)x, (int)(y + 1), seed), SmoothNoise((int)(x + 1), (int)(y + 1), seed), x - ((int)(x)));
// interpolate according to Y, parameter is fraction part of y
return CosineInterpolation(i1, i2, y - (int)y);
}

// calculates perlin noise for all iterations without using buffer
// interpData: interpolated data 
__global__ void CalcPerlinNonPrebuff(float* interpData, Perlinfo* info){

	// calculate coordinate attributes (index, size of cell, offset for this thread)
	int index = threadIdx.x + blockIdx.x*blockDim.x;

	// size of each thread-block cell
	int cellSizeX = round((1.0f*info->width) / blockDim.x);
	int cellSizeY = round((1.0f*info->height) / gridDim.x);

	for (int octave = 0; octave < info->octaves; octave++){

		// calculate actual frequency and amplitude
		float frequency = info->zoom*powf(2, octave);
		float amplitude = powf(info->persistence, octave);

		// offset for this thread
		int xOffset = threadIdx.x * cellSizeX;
		int yOffset = blockIdx.x * cellSizeY;

		// =================== PHASE 1: compute raw data

		for (int i = xOffset; i < xOffset + cellSizeX; i++){
			for (int j = yOffset; j < yOffset + cellSizeY; j++){

				// check out of bounds
				if (i < info->width && j < info->height){
					// calculate interpolated value
					float total = Interpolate(i * frequency, j * frequency, info->seed) * amplitude;
					// set value for new iteration
					interpData[i + j*info->width] += total;
				}
			}
		}
	}		
}


// ============================================= VARIATION 2:: COMPUTE RAW DATA AND USE THEM FOR INTERPOLATION ===============================================


// gets 1D coordinate from 2D coordinates; returns -1 if coordinates are outside the box
__device__ int Calc1D(int x, int y, int width, int height){
	int output = x + y*width;
	if (output >= width*height || x >= width || y >= height || y < 0 || x < 0) output = -1;
	return output;
}

// gets perlin noise from buffer if it has been already calculated
// otherwise it takes the value directly from noise algorithm
__device__ float GetNoise(float* data, int x, int y, float frequency, Perlinfo* info){

	// get 1D coordinate
	int coord = Calc1D(x, y, info->width, info->height);
	float output = data[coord];

	// if value hasn't been calculated or coordinates are outside of already calculated box,
	// take the value using the algorithm
	if (output == 0 || coord == -1) return Noise(x*frequency, y*frequency, info->seed);
	else return output;
}

// calculates offset according to the frequency
// isForward declares forward offset; otherwise it will calculate backward offset
__device__ int CalcOffset(int index, float frequency, bool isForward){
	int coord = index*frequency;
	float fCoord = index*frequency;

	// the next integer index that, according to the frequency multiplier, is one value behind the original
	return isForward ? (coord + 1 - fCoord + frequency) / frequency : (fCoord - coord + frequency) / frequency;
}

// calculates simple 3x3 smooth
__device__ float SmoothBuff(float* data, int x, int y, float frequency, Perlinfo* info){

	// calculate frequency offsets
	int offsetXP = CalcOffset(x, frequency, true);
	int offsetXM = CalcOffset(x, frequency, false);
	int offsetYP = CalcOffset(y, frequency, true);
	int offsetYM = CalcOffset(y, frequency, false);

	// calculate values and return their additions
	float corners = (GetNoise(data, x - offsetXM, y + offsetYP, frequency, info) + GetNoise(data, x - offsetXM, y - offsetYM, frequency, info) + GetNoise(data, x + offsetXP, y + offsetYP, frequency, info) + GetNoise(data, x + offsetXP, y - offsetYM, frequency, info)) / 16;
	float sides = (GetNoise(data, x - offsetXM, y, frequency, info) + GetNoise(data, x + offsetXP, y, frequency, info) + GetNoise(data, x, y + offsetYP, frequency, info) + GetNoise(data, x, y - offsetYM, frequency, info)) / 8;
	float ctr = GetNoise(data, x, y, frequency, info) / 4;

	return corners + sides + ctr;
}

// calculates interpolated noise, using cosine transform and 3x3 smooth algorithm
__device__ float InterpolateBuff(float* data, int x, int y, float frequency, Perlinfo* info){

	float i1, i2;

	// calculate fraction parts of coordinates
	float fractionX = (x*frequency) - (int)(x*frequency);
	float fractionY = (y*frequency) - (int)(y*frequency);

	// calculate frequency offsets
	int offsetXP = CalcOffset(x, frequency, true);
	int offsetXM = CalcOffset(x, frequency, false);
	int offsetYP = CalcOffset(y, frequency, true);
	int offsetYM = CalcOffset(y, frequency, false);

	i1 = CosineInterpolation(SmoothBuff(data, x, y, frequency, info), SmoothBuff(data, x + offsetXP, y, frequency, info), fractionX);
	i2 = CosineInterpolation(SmoothBuff(data, x, y + offsetYP, frequency, info), SmoothBuff(data, x + offsetXP, y + offsetYP, frequency, info), fractionX);

	return CosineInterpolation(i1, i2, fractionY);
}

// calculates perlin noise for one iteration
// interpData: interpolated data 
// rawData: raw data
__global__ void CalcPerlinPrebuff(float* interpData, float* rawData, Perlinfo* info, int iteration){

	// calculate actual frequency and amplitude
	float frequency = info->zoom*powf(2, iteration);
	float amplitude = powf(info->persistence, iteration);

	// calculate coordinate attribtues (index, size of cell, offset for this thread)
	int index = threadIdx.x + blockIdx.x*blockDim.x; 

	// size of each thread-block cell
	int cellSizeX = round((1.0f*info->width) / blockDim.x);
	int cellSizeY = round((1.0f*info->height) / gridDim.x);

	// offset for this thread
	int xOffset = threadIdx.x * cellSizeX;
	int yOffset = blockIdx.x * cellSizeY;

	// =================== PHASE 1: compute raw data

	for (int i = xOffset; i<xOffset + cellSizeX; i++){
		for (int j = yOffset; j<yOffset + cellSizeY; j++){

			// check out of bounds
			if (j < info->height && i < info->width){
				// if frequency is too low, next value will be the same as the previous one
				if (j != yOffset && (((int)(j*frequency)) == ((int)((j - 1)*frequency)))){
					rawData[i + j*info->width] = rawData[i + (j - 1)*info->width];
				}
				else{
					// compute perlin noise
					rawData[i + j*info->width] = Noise(i*frequency, j*frequency, info->seed);
				}
			}
		}
	}

	__syncthreads();

	// =================== PHASE 2: compute interpolated data

	for (int i = xOffset; i<xOffset + cellSizeX; i++){
		for (int j = yOffset; j<yOffset + cellSizeY; j++){

			// check out of bounds
			if (i < info->width && j < info->height){
				// calculate interpolated value
				float total = InterpolateBuff(rawData, i, j, frequency, info)*amplitude;
				// set value for new iteration
				if (iteration == 0) interpData[i + j*info->width] = total;
				else interpData[i + j*info->width] += total;
			}
		}
	}
}

// ==================================================================================================================================================

clock_t startTime;
clock_t startAllocTime;
clock_t endAllocTime;
clock_t startCalcTime;
clock_t endCalcTime;
clock_t startBmpTime;
clock_t endBmpTime;

int main(int argc, char** argv)
{
	startTime = clock();

	if (argc < 10) {
		std::cout << "Usage: " << argv[0] << " width height persistence octaves zoom seed outputImg blocks threads-per-block useBuffer{1,0}" << std::endl;
		exit(1);
	}



	Perlinfo info;
	
	info.width = atoi(argv[1]);
	info.height = atoi(argv[2]);
	info.persistence = atof(argv[3]);
	info.octaves = atoi(argv[4]);
	info.zoom = atof(argv[5]);
	info.seed = atoi(argv[6]);
	char* outputImg  = (char*)malloc(sizeof(char) * strlen(argv[7]));
	strcpy(outputImg, argv[7]);
	int blocks = atoi(argv[8]);
	int threads = atoi(argv[9]);
	bool useBuffer = argc < 10 || atoi(argv[10]) == 1;
	

	/*info.width = 512;
	info.height = 512;
	info.persistence = .75f;
	info.octaves = 16;
	info.zoom = 0.03f;
	info.seed = 511;
	char* outputImg = "outputx.bmp";
	int blocks = 128;
	int threads = 64;
	bool useBuffer = false;*/
	
	unsigned char *bitmap  = (unsigned char *)malloc(3*info.width*info.height*sizeof(unsigned char));

	// write info about GPU
	CUDA_SAFE(CUDAWriteInfo(info));

	CUDA_SAFE(cudaSetDevice(0));

	// calculate bitmap
	CUDA_SAFE(CUDAPerlinNoise(info, bitmap, blocks, threads, useBuffer));

	// reset (because of nsight)
	CUDA_SAFE(cudaDeviceReset());


	// init bitmap decoder and write bitmap
	BmpDecoder decoder(info.width, info.height);
	decoder.SaveAsFile(bitmap, outputImg);

	endBmpTime = clock();

	WriteResult(info, threads,blocks,useBuffer);

	return 0;
}


cudaError_t CUDAWriteInfo(Perlinfo info){

	int count;
	cudaGetDeviceCount(&count);

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	
	printf("Number of devices: %d\n", count);
	printf("GPU name: %s\n", prop.name);
	printf("Memory per block:%d\n", prop.sharedMemPerBlock);
	printf("Multi CPU count:%d\n", prop.multiProcessorCount);
	printf("Warp size: %d\n", prop.warpSize);
	printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);

	printf("=====  USER SETTINGS =====\n");
	printf("Size of image: %d x %d \n",info.width,info.height);
	printf("Number of iterations: %d \n",info.octaves);
	printf("Persistence: %.4f \n", info.persistence);
	printf("Zoom: %.4f\n", info.zoom);
	printf("Seed: %d\n", info.seed);
	
	return cudaGetLastError();
}

// calculates Perlin noise and returns bitmap data
cudaError_t CUDAPerlinNoise(Perlinfo info, unsigned char* outputBitmap, int blocks, int threads, bool useBuffer)
{
	startAllocTime = clock();

	// allocate buffers and struct with settings
	float* interpData = 0;
	float* rawData = 0;
	Perlinfo* infoC;

	// interpolated bitmap
	CUDA_SAFE(cudaMalloc((void**)&interpData,  info.width*info.height*sizeof(float)));
	CUDA_SAFE(cudaMalloc((void**)&infoC,  sizeof(Perlinfo)));
	CUDA_SAFE(cudaMemcpy(infoC, &info, sizeof(Perlinfo), cudaMemcpyHostToDevice));
	if (useBuffer){
	  CUDA_SAFE(cudaMalloc((void**)&rawData, info.width*info.height*sizeof(float)));
	}

	endAllocTime = clock();

	startCalcTime = clock();

	if (useBuffer){
		// main loop -> for each octave, calculate perlin noise
		for (int i = 0; i < info.octaves; i++){

			int dimBlock = threads;
			int dimGrid = blocks;

			// reset raw data
			CUDA_SAFE(cudaMemset(rawData, 0x00, info.width*info.height*sizeof(float)));
			
			// calculate perlin noise for one iteration
			CalcPerlinPrebuff<<<dimGrid, dimBlock >>>(interpData, rawData, infoC, i);

			CUDA_SAFE(cudaGetLastError());

			// wait for device
			CUDA_SAFE(cudaDeviceSynchronize());
		}
	}
	else{
		int dimBlock = threads;
		int dimGrid = blocks;

		// reset interpolated data
		CUDA_SAFE(cudaMemset(interpData, 0x00, info.width*info.height*sizeof(float)));

		// calculate perlin noise for all iterations
		CalcPerlinNonPrebuff<<<dimGrid, dimBlock >>>(interpData, infoC);

		CUDA_SAFE(cudaGetLastError());

		// wait for device
		CUDA_SAFE(cudaDeviceSynchronize());
	}

	endCalcTime = clock();

	// copy GPU interpolated data to the host memory and copy them to the output bitmap
	float *interpDataHost  = (float *)malloc(info.width*info.height*sizeof(float));
	CUDA_SAFE(cudaMemcpy(interpDataHost, interpData, info.width*info.height*sizeof(float), cudaMemcpyDeviceToHost));

	startBmpTime = clock();
	for(int i=0; i<info.width; i++){
		for(int j=0; j<info.height; j++){

			// get final perlin value for this pixel
			float total = interpDataHost[i+j*info.width];

			// fix bound values
			if(total < -1) total = -1;
			if(total > 1) total = 1;

			// set bitmap
			int colorVal = (total+1)*127;

			outputBitmap[(i+j*info.width)*3+2] = (unsigned char)(colorVal);
			outputBitmap[(i+j*info.width)*3+1] = (unsigned char)(colorVal);
			outputBitmap[(i+j*info.width)*3+0] = (unsigned char)(colorVal);
		}
	}

	return cudaGetLastError();
}

void WriteResult(Perlinfo info, int threads, int blocks, bool buffer){
	float totalTime = float( clock () - startTime );
	float allocTime = float( endAllocTime - startAllocTime );
	float calcTime = float( endCalcTime - startCalcTime );
	float bmpTime = float( endBmpTime - startBmpTime );

	//printf("\nPerlin calculation finished...\n");

	printf("\%dx%d - %d for %dx%d, buffer=%d\n",info.width,info.height,info.octaves,threads,blocks,buffer ? 1 : 0);
	//printf("Allocation time:	%.f ms \n", allocTime);
	printf("Calculation time:	%.f ms \n", calcTime);
	//printf("BMP write time:		%.f ms \n", bmpTime);
	printf("---------------------------\n");
	//printf("Total time:		%.f ms \n", totalTime);
}