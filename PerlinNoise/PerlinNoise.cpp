// PerlinNoise.cpp : Defines the entry point for the console application.
//
#include <iostream>
#include <fstream>
#include <iosfwd>
#include <math.h> 

using namespace std;
#include "stdafx.h"
#include <time.h>


// perlin noise generator
class PerlinNoise{
private:
	int seed;
public:

	PerlinNoise(int seed) : seed(seed){}


	// calculates perlin noise at specified coordinates, returns value between -1 and 1
	float PerlinNoise2D(int x, int y, float persistence, int octaves, float zoom)
	{
		// total output noise value
		float total = 0.0f;
		// initial frequency
		float frequency = zoom;
		// initial amplitude
		float amplitude = 1.0f;

		// for each octave
		for (int i = 0; i < octaves; i++)
		{
			// calculate noise
			total = total + InterpolatedNoise(x * frequency, y * frequency) * amplitude;
			// set frequency and amplitude
			frequency = frequency * 2;
			// decrease amplitude
			amplitude = amplitude * persistence;

		}

		// fix bound values
		if(total < -1) total = -1;
		if(total > 1) total = 1;
		return total;
	}

	void PerlinNoise2DPlane(unsigned char* data, int width, int height, float persistence, int octaves, float zoom){
		// for each pixel, generate its perlin value and write it into RGB channel
		for(int i=0; i<width; i++)
		{
			for(int j=0; j<height; j++)
			{
				float val = this->PerlinNoise2D(i,j,persistence,octaves, zoom);
				float colorVal = (val+1)*127;

				data[(i+j*width)*3+2] = (unsigned char)(colorVal);
				data[(i+j*width)*3+1] = (unsigned char)(colorVal);
				data[(i+j*width)*3+0] = (unsigned char)(colorVal);
			}
		}
	}

private:
	// gets interpolated noise for various levels of zooming
	float InterpolatedNoise(float x, float y)
	{
		float i1, i2;
		// interpolate according to X for bottom part, parameter is fraction part of x
		i1 = CosineInterpolation(SmoothNoise((int)x, (int)y), SmoothNoise((int)(x + 1), (int)(y)), x - ((int)(x)));
		// interpolate according to X for upper part
		i2 = CosineInterpolation(SmoothNoise((int)x, (int)(y + 1)), SmoothNoise((int)(x + 1), (int)(y + 1)), x - ((int)(x)));
		// interpolate according to Y, parameter is fraction part of y
		return CosineInterpolation(i1, i2, y - (int)y);
	}

	// linear coherent-noise function
	float Noise(int x, int y)
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

	float SmoothNoise(int x, int y)
	{
		/*
		*  |x-1,y+1| x ,y+1|x+1,y+1|
		*  |x-1,y  | x ,y  |x+1,y  |
		*  |x-1,y-1| x ,y-1|x+1,y-1|
		*/

		// smooth corners (left upper, left down etc.)
		float corners = (Noise(x - 1, y - 1) + Noise(x + 1, y - 1) + Noise(x - 1, y + 1) + Noise(x + 1, y + 1)) / 16;
		// smooth sides (left, right,...)
		float sides = (Noise(x - 1, y) + Noise(x + 1, y) + Noise(x, y - 1) + Noise(x, y + 1)) / 8;
		// smooth center
		float center = Noise(x, y) / 4;
		// sum values
		return corners + sides + center;
	}

	// cosine interpolation
	// v1: first value to interpolate between
	// v2: second value to interpolate between
	// x: interpolating parameter
	float CosineInterpolation(float v1, float v2, float x)
	{
		float f = (float)(1 - cos(x * 3.1415927)) * 0.5f;
		return v1 * (1 - f) + v2 * f;
	}
};


// bitmap decoder
class BmpDecoder{
private:
	int width;
	int height;
public:
	BmpDecoder(int width, int height) : width(width), height(height){

	}

	// decodes
	void SaveAsFile(unsigned char* data, const char* filename){

		FILE *f;

		// 54 bytes header + number of pixels * number of channels
		int filesize = 14 + 40 + 3 * width*height;

		// 14 bytes for header
		unsigned char bmpfileheader[14] = { 'B', 'M', 0, 0, 0, 0, 0, 0, 0, 0, 54, 0, 0, 0 };
		// 40 bytes (version from Windows 3.1x - without alpha channel, color space and ICC profiles)
		unsigned char bmpinfoheader[40] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 24, 0 };

		// size of each row is rounded up to a multiple of 4B by padding
		unsigned char bmppad[3] = { 0, 0, 0 };

		// file size, 4B
		bmpfileheader[2] = (unsigned char)(filesize);
		bmpfileheader[3] = (unsigned char)(filesize >> 8);
		bmpfileheader[4] = (unsigned char)(filesize >> 16);
		bmpfileheader[5] = (unsigned char)(filesize >> 24);

		// header size, 1B
		bmpinfoheader[0] = (unsigned char)(40);
		// width in pixels, 4B
		bmpinfoheader[4] = (unsigned char)(width);
		bmpinfoheader[5] = (unsigned char)(width >> 8);
		bmpinfoheader[6] = (unsigned char)(width >> 16);
		bmpinfoheader[7] = (unsigned char)(width >> 24);
		// height in pixels, 4B
		bmpinfoheader[8] = (unsigned char)(height);
		bmpinfoheader[9] = (unsigned char)(height >> 8);
		bmpinfoheader[10] = (unsigned char)(height >> 16);
		bmpinfoheader[11] = (unsigned char)(height >> 24);

		// write data into bmp file
		f = fopen(filename, "wb");
		fwrite(bmpfileheader, 1, 14, f);
		fwrite(bmpinfoheader, 1, 40, f);
		for (int i = 0; i < height; i++)
		{
			// write one row and pad to have size of each row a multiple of 4B
			fwrite(data + (width*(height - i - 1) * 3), 3, width, f);
			fwrite(bmppad, 1, (4 - (width * 3) % 4) % 4, f);
		}
		fclose(f);
	}
};




unsigned char* data;

// =================================================================================
clock_t startTime;
clock_t startAllocTime;
clock_t endAllocTime;
clock_t startCalcTime;
clock_t endCalcTime;
clock_t startBmpTime;
clock_t endBmpTime;

void WriteResult(){
	float totalTime = float(clock() - startTime);
	float allocTime = float(endAllocTime - startAllocTime);
	float calcTime = float(endCalcTime - startCalcTime);
	float bmpTime = float(endBmpTime - startBmpTime);

	printf("\nPerlin calculation finished...\n");
	printf("Allocation time:	%.f ms \n", allocTime);
	printf("Calculation time:	%.f ms \n", calcTime);
	printf("BMP write time:		%.f ms \n", bmpTime);
	printf("---------------------------\n");
	printf("Total time:		%.f ms \n", totalTime);
}

int main(int argc, char ** argv) 
{
	startTime = clock();

   if (argc != 8) {
		cout << "Usage: " << argv[0] << " width height persistence octaves zoom seed outputImg" << endl;
		exit(1);
	}

	int width = atoi(argv[1]);
	int height = atoi(argv[2]);
	float persistence = atof(argv[3]);
	int octaves = atoi(argv[4]);
	float zoom = atof(argv[5]);
	int seed = atoi(argv[6]);
	char* outputImg  = (char*)malloc(sizeof(char) * strlen(argv[7]));
	strcpy(outputImg, argv[7]);

	/*int width = 128;
	int height = 128;
	float persistence = .9f;
	int octaves = 20;
	float zoom = .13f;
	int seed = 50025;
	char* outputImg = "outputfa.bmp";*/

	startAllocTime = clock();
	// allocate bitmap array
	if( data )
		free( data );
	data = (unsigned char *)malloc(3*width*height);

	// reset bites in data array
	memset(data,0,sizeof(data));
	endAllocTime = clock();
	startCalcTime = clock();

	// init perlin generator
	PerlinNoise noise(seed);
	// calculate perlin noise
	noise.PerlinNoise2DPlane(data, width, height, persistence, octaves, zoom);

	endCalcTime = clock();
	startBmpTime = clock();

	// init bitmap decoder
	BmpDecoder decoder(width, height);
	// write bitmap
	decoder.SaveAsFile(data, outputImg);
	delete[] data;

	endBmpTime = clock();
	WriteResult();

	return 0;
}

