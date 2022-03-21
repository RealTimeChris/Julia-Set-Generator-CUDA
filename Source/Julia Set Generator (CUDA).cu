// Julia Set Generator (CUDA).cu
// July 2018 (Updated September 2019)
// Generates and saves Julia Sets, with custom color gradients.
// Chris M
// https://github.com/RealTimeChris


#if !defined WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include <Windows.h>
#include <stdio.h>
#include <string.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuComplex.h"


// Resolution of the output image.
#define HEIGHT 4320
#define WIDTH 7680


// Collects the real and imaginary bounds of a section of the complex plane.
void planeCollect(double *realLeft, double *realRight, double *imagUpper, double *imagLower)
{
	char realL[256], realR[256], imagU[256], imagL[256];

	printf("Enter the left bound of the complex plane: "); gets_s(realL);
	printf("Enter the right bound of the complex plane: "); gets_s(realR);
	printf("Enter the upper bound of the complex plane: "); gets_s(imagU);
	printf("Enter the lower bound of the complex plane: "); gets_s(imagL);

	sscanf_s(realL, "%lf", realLeft);
	sscanf_s(realR, "%lf", realRight);
	sscanf_s(imagU, "%lf", imagUpper);
	sscanf_s(imagL, "%lf", imagLower);

	return;
}


// Calculates the real and imaginary delta per pixel along each axis.
void intervalCalc(double realLeft, double realRight, double imagUpper,double imagLower, double* realYInt, double* imagXInt)
{
	*realYInt = (realRight - realLeft) / (WIDTH - 1);
	*imagXInt = (imagUpper - imagLower) / (HEIGHT - 1);

	return;
}


// Collects the c value for use in: z_n = z^2 + c
void complexGet(cuDoubleComplex* c)
{
	char aRaw[256], bRaw[256];
	double a, b;

	printf("\nEnter the real part of c: "); gets_s(aRaw);
	printf("Enter the imaginary part of c: "); gets_s(bRaw);

	sscanf_s(aRaw, "%lf", &a);
	sscanf_s(bRaw, "%lf", &b);

	*c = make_cuDoubleComplex(a, b);

	return;
}


// Defines the x and y coordinates (i,j) of the current pixel that is being processed, using the current block and thread values.
// This is where 2D of 2D indexing is abstracted into global 2D indexing.
__device__ void pixelIndex(int* i, int* j)
{
	*i = (blockIdx.x * blockDim.x) + threadIdx.x;
	*j = (blockIdx.y * blockDim.y) + threadIdx.y;

	return;
}


// Calculates where in the row-major linearized array to store the current value.
// This is where the global 2D indexing is abstracted into global 1D indexing.
__device__ void matrixIndex(int i, int j, int *matrixLoc)
{
	*matrixLoc = (i * WIDTH) + j;

	return;
}


// Calculates the real-interval for the current pixel.
__device__ void realCV(double realLeftDev, double realYIntDev, int j, double *currentVal)
{
	*currentVal = realLeftDev + (realYIntDev * j);

	return;
}


// Calculates the imaginary-interval value for the current pixel.
__device__ void imagCV(double imagUpperDev, double imagXIntDev, int i, double *currentVal)
{
	*currentVal = imagUpperDev - (imagXIntDev * i);

	return;
}


// Fills the elements of a matrix that represents the complex plane.
__global__ void complexFill(
	double *realLeftDev, double *realYIntDev, double *imagUpperDev,
	double *imagXIntDev, cuDoubleComplex *complexMatrixDev)
{
	int i, j, matrixLoc;
	double realCurrentVal, imagCurrentVal;
	cuDoubleComplex complexCurrentVal;

	pixelIndex(&i, &j);

	realCV(*realLeftDev, *realYIntDev, j, &realCurrentVal);
	imagCV(*imagUpperDev, *imagXIntDev, i, &imagCurrentVal);

	complexCurrentVal = make_cuDoubleComplex(realCurrentVal, imagCurrentVal);

	matrixIndex(i, j, &matrixLoc);

	complexMatrixDev[matrixLoc] = complexCurrentVal;

	return;
}


// Tracks the number of iterations each pixel takes to diverge.
__device__ void divCheck(cuDoubleComplex zN, int *divCountDev, int matrixLoc)
{
	int notDiverged;

	notDiverged = cuCabs(zN) < 2;

	divCountDev[matrixLoc] = (divCountDev[matrixLoc]) + notDiverged;

	return;
}


// Fills the elements of a matrix with the results of z_n = z^2 + c, and executes divergence-counting.
__global__ void zNCalc(cuDoubleComplex *zMatrix, cuDoubleComplex *c, int *divCountDev)
{

	for (int iter = 0; iter < 100; iter++)
	{
		int i, j, matrixLoc;
		cuDoubleComplex zN, z;

		pixelIndex(&i, &j);

		matrixIndex(i, j, &matrixLoc);

		z = zMatrix[matrixLoc];

		zN = cuCadd(cuCmul(z, z), *c);

		divCheck(zN, divCountDev, matrixLoc);

		zMatrix[matrixLoc] = zN;
	}

	return;
}


// Collects the a,k,c, and d transform values for each color layer.
void gradientCollect(
	float *redD, float *redK, float *redA, float *redC,
	float *greenD, float *greenK, float *greenA, float *greenC,
	float *blueD, float *blueK, float *blueA, float *blueC)
{
	char redDS[256], redKS[256], redAS[256], redCS[256];
	char greenDS[256], greenKS[256], greenAS[256], greenCS[256];
	char blueDS[256], blueKS[256], blueAS[256], blueCS[256];

	printf("COLOR GRADIENT DESIGN:\n");

	printf("Enter d (red): "); gets_s(redDS);
	printf("Enter k (red): "); gets_s(redKS);
	printf("Enter a (red): "); gets_s(redAS);
	printf("Enter c (red): "); gets_s(redCS);

	printf("\nEnter d (green): "); gets_s(greenDS);
	printf("Enter k (green): "); gets_s(greenKS);
	printf("Enter a (green): "); gets_s(greenAS);
	printf("Enter c (green): "); gets_s(greenCS);

	printf("\nEnter d (blue): "); gets_s(blueDS);
	printf("Enter k (blue): "); gets_s(blueKS);
	printf("Enter a (blue): "); gets_s(blueAS);
	printf("Enter c (blue): "); gets_s(blueCS);

	sscanf_s(redDS, "%f", redD);
	sscanf_s(redKS, "%f", redK);
	sscanf_s(redAS, "%f", redA);
	sscanf_s(redCS, "%f", redC);

	sscanf_s(greenDS, "%f", greenD);
	sscanf_s(greenKS, "%f", greenK);
	sscanf_s(greenAS, "%f", greenA);
	sscanf_s(greenCS, "%f", greenC);

	sscanf_s(blueDS, "%f", blueD);
	sscanf_s(blueKS, "%f", blueK);
	sscanf_s(blueAS, "%f", blueA);
	sscanf_s(blueCS, "%f", blueC);

	return;
}


// Transforms the divergence-count matrix into a single color layer for the RGB matrix output.
__global__ void colorTransform(
	int *divCountDev, unsigned char *colorLayerDev, float *colorDDev,
	float *colorKDev, float *colorADev, float *colorCDev)
{
	int i, j, matrixLoc;
	float colorValue;

	pixelIndex(&i, &j);

	matrixIndex(i, j, &matrixLoc);

	colorValue = divCountDev[matrixLoc];

	if (colorValue == 100)
	{
		colorValue = 0;
	}
	else
	{
		colorValue = colorValue - (*colorDDev);
		colorValue = (*colorKDev) * colorValue;
		colorValue = sinf(colorValue);
		colorValue = (*colorADev) * colorValue;
		colorValue = colorValue + (*colorCDev);
		colorValue = roundf(colorValue);

		if (colorValue > 255)
		{
			colorValue = 255;
		}
		else if (colorValue < 0)
		{
			colorValue = 0;
		}
	}

	colorLayerDev[matrixLoc] = (unsigned char)colorValue;

	return;
}


// Saves the row-major linearized RGB arrays as a bitmap image on the Windows Desktop.
void bmpSave(unsigned char *redLayer, unsigned char *greenLayer, unsigned char *blueLayer)
{
	char *pathUser;
	pathUser = (char *)malloc(strlen(getenv("USERPROFILE")));
	pathUser = { getenv("USERPROFILE") };

	char pathDesktop[12] = { "\\Desktop\\" };

	char pathFName[256];
	printf("Enter a file name: "); gets_s(pathFName);

	char pathFExt[5] = { ".bmp" };

	char *filePath;
	filePath = (char *)malloc(strlen(getenv("USERPROFILE")) + strlen(pathDesktop) + strlen(pathFName) + strlen(pathFExt) + 1);

	strcpy(filePath, pathUser);
	strcat(filePath, pathDesktop);
	strcat(filePath, pathFName);
	strcat(filePath, pathFExt);

	unsigned char bmpFileHeader[14] = { 'B','M', 0,0,0,0, 0,0, 0,0, 54,0,0,0 };
	unsigned char bmpFileInfoHeader[40] = { 40,0,0,0, 0,0,0,0, 0,0,0,0, 1,0, 24,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0 };

	int fileSize = (54 + (3 * HEIGHT * WIDTH));

	bmpFileHeader[2] = (unsigned char)(fileSize);
	bmpFileHeader[3] = (unsigned char)(fileSize >> 8);
	bmpFileHeader[4] = (unsigned char)(fileSize >> 16);
	bmpFileHeader[5] = (unsigned char)(fileSize >> 24);

	bmpFileInfoHeader[4] = (unsigned char)(WIDTH);
	bmpFileInfoHeader[5] = (unsigned char)(WIDTH >> 8);
	bmpFileInfoHeader[6] = (unsigned char)(WIDTH >> 16);
	bmpFileInfoHeader[7] = (unsigned char)(WIDTH >> 24);
	bmpFileInfoHeader[8] = (unsigned char)(HEIGHT);
	bmpFileInfoHeader[9] = (unsigned char)(HEIGHT >> 8);
	bmpFileInfoHeader[10] = (unsigned char)(HEIGHT >> 16);
	bmpFileInfoHeader[11] = (unsigned char)(HEIGHT >> 24);

	unsigned char *bmpFileComplete;
	bmpFileComplete = (unsigned char *)malloc(fileSize * sizeof(unsigned char));

	for (int i = 0; i < 14; i++) bmpFileComplete[i] = bmpFileHeader[i];
	for (int i = 0; i < 40; i++) bmpFileComplete[14 + i] = bmpFileInfoHeader[i];

	// Flip the color layers along the x-axis of each matrix in accordance with the bitmap file format.
	unsigned char *redLayerFlip, *greenLayerFlip, *blueLayerFlip;
	redLayerFlip = (unsigned char *)malloc((HEIGHT * WIDTH) * sizeof(unsigned char));
	greenLayerFlip = (unsigned char *)malloc((HEIGHT * WIDTH) * sizeof(unsigned char));
	blueLayerFlip = (unsigned char *)malloc((HEIGHT * WIDTH) * sizeof(unsigned char));

	for (int i = 0; i < HEIGHT; i++)
	{
		for (int j = 0; j < WIDTH; j++)
		{
			redLayerFlip[(HEIGHT - 1 - i) * WIDTH + j] = redLayer[i * WIDTH + j];
			greenLayerFlip[(HEIGHT - 1 - i) * WIDTH + j] = greenLayer[i * WIDTH + j];
			blueLayerFlip[(HEIGHT - 1 - i) * WIDTH + j] = blueLayer[i * WIDTH + j];
		}
	}

	printf("\nSaving the image to desktop...");
	for (int i = 0; i < (3 * HEIGHT * WIDTH); i += 3)
	{
		bmpFileComplete[(54 + i) + 2] = redLayerFlip[i / 3];
		bmpFileComplete[(54 + i) + 1] = greenLayerFlip[i / 3];
		bmpFileComplete[(54 + i) + 0] = blueLayerFlip[i / 3];
	}

	FILE *file;
	file = fopen(filePath, "wb+");

	fwrite(bmpFileComplete, sizeof(unsigned char), (54 + (3 * HEIGHT * WIDTH)), file);

	fclose(file);
	printf(" Done!\n\n");

	return;
}


// Main loop for the console program.
int main(void)
{
	// Welcome message.
	printf("Julia Set Generator (CUDA)\nCreated by Chris M\nhttps://github.com/RealTimeChris\n\n");
	printf("WARNING: There is no data validation or exception handling, so watch your input to avoid undefined behavior.\n\n");

	while (1 == 1)
	{
		// Grid and block values.
		dim3 threadsPB(8, 128);
		dim3 grid(HEIGHT / threadsPB.x, WIDTH / threadsPB.y);

		// Host copies of user input.
		double realLeft, realRight, realYInt, imagUpper, imagLower, imagXInt;
		cuDoubleComplex c;

		// Collect and prepare user input.
		printf("JULIA SET PARAMETERS:\n");
		planeCollect(&realLeft, &realRight, &imagUpper, &imagLower);
		complexGet(&c);
		intervalCalc(realLeft, realRight, imagUpper, imagLower, &realYInt, &imagXInt);

		// Device copies of the necessary values.
		printf("\nGetting the GPU warmed up...");
		double* realLeftDev;
		cudaMalloc((void **)&realLeftDev, sizeof(double));
		cudaMemcpy(realLeftDev, &realLeft, sizeof(double), cudaMemcpyHostToDevice);

		double* realYIntDev;
		cudaMalloc((void **)&realYIntDev, sizeof(double));
		cudaMemcpy(realYIntDev, &realYInt, sizeof(double), cudaMemcpyHostToDevice);

		double* imagUpperDev;
		cudaMalloc((void **)&imagUpperDev, sizeof(double));
		cudaMemcpy(imagUpperDev, &imagUpper, sizeof(double), cudaMemcpyHostToDevice);

		double* imagXIntDev;
		cudaMalloc((void **)&imagXIntDev, sizeof(double));
		cudaMemcpy(imagXIntDev, &imagXInt, sizeof(double), cudaMemcpyHostToDevice);

		cuDoubleComplex* cDev;
		cudaMalloc((void **)&cDev, sizeof(cuDoubleComplex));
		cudaMemcpy(cDev, &c, sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

		// Device copy of the complex matrix.
		cuDoubleComplex *complexMatrixDev;
		cudaMalloc((void **)&complexMatrixDev, (HEIGHT * WIDTH) * sizeof(cuDoubleComplex));
		printf(" Done!\n\n");

		// Fill the complex matrix on the device.
		printf("Creating the complex plane...");
		complexFill << <grid, threadsPB >> > (realLeftDev, realYIntDev, imagUpperDev, imagXIntDev, complexMatrixDev);
		cudaDeviceSynchronize();
		printf(" Done!\n\n");

		// Free up some device memory.
		cudaFree(realLeftDev);
		cudaFree(realYIntDev);
		cudaFree(imagUpperDev);
		cudaFree(imagXIntDev);

		// Device copy of divergence-count matrix.
		int *divCountDev;
		cudaMalloc((void **)&divCountDev, (HEIGHT * WIDTH) * sizeof(int));

		// Fill the divergence-count matrix by executing the iterations of the function.
		printf("Executing 100 iterations of 33.18 million instances of z_n = z^2 + c...");
		zNCalc << <grid, threadsPB >> > (complexMatrixDev, cDev, divCountDev);
		cudaDeviceSynchronize();
		printf(" Done!\n\n");

		// Free up some device memory.
		cudaFree(complexMatrixDev);
		cudaFree(cDev);

		// Host copies of variables for the color gradient.
		float redD, redK, redA, redC;
		float greenD, greenK, greenA, greenC;
		float blueD, blueK, blueA, blueC;

		// Collect the transform values for creating the color gradient.
		gradientCollect(
			&redD, &redK, &redA, &redC,
			&greenD, &greenK, &greenA, &greenC,
			&blueD, &blueK, &blueA, &blueC);

		// Device copies of the color gradient transform values.
		float *redDDev, *redKDev, *redADev, *redCDev;
		float *greenDDev, *greenKDev, *greenADev, *greenCDev;
		float *blueDDev, *blueKDev, *blueADev, *blueCDev;

		cudaMalloc((void **)&redDDev, sizeof(float));
		cudaMemcpy(redDDev, &redD, sizeof(float), cudaMemcpyHostToDevice);

		cudaMalloc((void **)&redKDev, sizeof(float));
		cudaMemcpy(redKDev, &redK, sizeof(float), cudaMemcpyHostToDevice);

		cudaMalloc((void **)&redADev, sizeof(float));
		cudaMemcpy(redADev, &redA, sizeof(float), cudaMemcpyHostToDevice);

		cudaMalloc((void **)&redCDev, sizeof(float));
		cudaMemcpy(redCDev, &redC, sizeof(float), cudaMemcpyHostToDevice);

		cudaMalloc((void **)&greenDDev, sizeof(float));
		cudaMemcpy(greenDDev, &greenD, sizeof(float), cudaMemcpyHostToDevice);

		cudaMalloc((void **)&greenKDev, sizeof(float));
		cudaMemcpy(greenKDev, &greenK, sizeof(float), cudaMemcpyHostToDevice);

		cudaMalloc((void **)&greenADev, sizeof(float));
		cudaMemcpy(greenADev, &greenA, sizeof(float), cudaMemcpyHostToDevice);

		cudaMalloc((void **)&greenCDev, sizeof(float));
		cudaMemcpy(greenCDev, &greenC, sizeof(float), cudaMemcpyHostToDevice);

		cudaMalloc((void **)&blueDDev, sizeof(float));
		cudaMemcpy(blueDDev, &blueD, sizeof(float), cudaMemcpyHostToDevice);

		cudaMalloc((void **)&blueKDev, sizeof(float));
		cudaMemcpy(blueKDev, &blueK, sizeof(float), cudaMemcpyHostToDevice);

		cudaMalloc((void **)&blueADev, sizeof(float));
		cudaMemcpy(blueADev, &blueA, sizeof(float), cudaMemcpyHostToDevice);

		cudaMalloc((void **)&blueCDev, sizeof(float));
		cudaMemcpy(blueCDev, &blueC, sizeof(float), cudaMemcpyHostToDevice);

		// Device copy of the red layer.
		unsigned char *redLayerDev;
		cudaMalloc((void **)&redLayerDev, (HEIGHT * WIDTH) * sizeof(unsigned char));

		// Create the red layer.
		printf("\nTransforming the red layer...");
		colorTransform << <grid, threadsPB >> > (divCountDev, redLayerDev, redDDev, redKDev, redADev, redCDev);
		cudaDeviceSynchronize();
		printf(" Done!\n");

		// Host copy of the red layer.
		unsigned char *redLayer;
		redLayer = (unsigned char *)malloc((HEIGHT * WIDTH) * sizeof(unsigned char));
		cudaMemcpy(redLayer, redLayerDev, (HEIGHT * WIDTH) * sizeof(unsigned char), cudaMemcpyDeviceToHost);

		// Free up some device memory.
		cudaFree(redLayerDev);
		cudaFree(redDDev);
		cudaFree(redKDev);
		cudaFree(redADev);
		cudaFree(redCDev);

		// Device copy of the green layer.
		unsigned char *greenLayerDev;
		cudaMalloc((void **)&greenLayerDev, (HEIGHT * WIDTH) * sizeof(unsigned char));

		// Create the green layer.
		printf("Transforming the green layer...");
		colorTransform << <grid, threadsPB >> > (divCountDev, greenLayerDev, greenDDev, greenKDev, greenADev, greenCDev);
		cudaDeviceSynchronize();
		printf(" Done!\n");

		// Host copy of the green layer.
		unsigned char *greenLayer;
		greenLayer = (unsigned char *)malloc((HEIGHT * WIDTH) * sizeof(unsigned char));
		cudaMemcpy(greenLayer, greenLayerDev, (HEIGHT * WIDTH) * sizeof(unsigned char), cudaMemcpyDeviceToHost);

		// Free up some device memory.
		cudaFree(greenLayerDev);
		cudaFree(greenDDev);
		cudaFree(greenKDev);
		cudaFree(greenADev);
		cudaFree(greenCDev);

		// Device copy of the blue layer.
		unsigned char *blueLayerDev;
		cudaMalloc((void **)&blueLayerDev, (HEIGHT * WIDTH) * sizeof(unsigned char));

		// Create the blue layer.
		printf("Transforming the blue layer...");
		colorTransform << <grid, threadsPB >> > (divCountDev, blueLayerDev, blueDDev, blueKDev, blueADev, blueCDev);
		cudaDeviceSynchronize();
		printf(" Done!\n\n");

		// Host copy of the blue layer.
		unsigned char *blueLayer;
		blueLayer = (unsigned char *)malloc((HEIGHT * WIDTH) * sizeof(unsigned char));
		cudaMemcpy(blueLayer, blueLayerDev, (HEIGHT * WIDTH) * sizeof(unsigned char), cudaMemcpyDeviceToHost);

		// Free up some device memory.
		cudaFree(blueLayerDev);
		cudaFree(blueDDev);
		cudaFree(blueKDev);
		cudaFree(blueADev);
		cudaFree(blueCDev);
		cudaFree(divCountDev);

		// Render the image to disk.
		bmpSave(redLayer, greenLayer, blueLayer);

		// Free up some host memory.
		free(redLayer);
		free(greenLayer);
		free(blueLayer);

		// Ask for another run.
		char progRepeat[12];
		printf("Would you like to create another? (y/n) "); gets_s(progRepeat);
		printf("\n");

		if (progRepeat[0] != 'y')
		{
			break;
		}
	}

	char progEnd[256];
	printf("Press enter to exit...");
	gets_s(progEnd);

	return 0;
}
