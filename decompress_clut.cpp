/*
 #
 #  File            : decompress_clut.cpp
 #                   ( C++ source file )
 #
 #  Description     : CLUT decompression algorithm from a set of keypoints.
 #                    This is an implementation of the research paper :
 #                    "An Efficient 3D Color LUT Compression Algorithm Based on a Multi-Scale Anisotropic Diffusion Scheme",
 #                    by D. Tschumperlé, C. Porquet and A. Mahboubi,
 #                    available at : https://hal.archives-ouvertes.fr/hal-02066484
 #
 #  Author          : David Tschumperle.
 #                    ( http://tschumperle.users.greyc.fr/ )
 #
 #  Licenses        : This file is 'dual-licensed', you have to choose one
 #                    of the two licenses below to apply.
 #
 #                    CeCILL-C
 #                    The CeCILL-C license is close to the GNU LGPL.
 #                    ( http://www.cecill.info/licences/Licence_CeCILL-C_V1-en.html )
 #
 #                or  CeCILL v2.1
 #                    The CeCILL license is compatible with the GNU GPL.
 #                    ( http://www.cecill.info/licences/Licence_CeCILL_V2.1-en.html )
 #
 #  This software is governed either by the CeCILL or the CeCILL-C license
 #  under French law and abiding by the rules of distribution of free software.
 #  You can  use, modify and or redistribute the software under the terms of
 #  the CeCILL or CeCILL-C licenses as circulated by CEA, CNRS and INRIA
 #  at the following URL: "http://www.cecill.info".
 #
 #  As a counterpart to the access to the source code and  rights to copy,
 #  modify and redistribute granted by the license, users are provided only
 #  with a limited warranty  and the software's author,  the holder of the
 #  economic rights,  and the successive licensors  have only  limited
 #  liability.
 #
 #  In this respect, the user's attention is drawn to the risks associated
 #  with loading,  using,  modifying and/or developing or reproducing the
 #  software by the user in light of its specific status of free software,
 #  that may mean  that it is complicated to manipulate,  and  that  also
 #  therefore means  that it is reserved for developers  and  experienced
 #  professionals having in-depth computer knowledge. Users are therefore
 #  encouraged to load and test the software's suitability as regards their
 #  requirements in conditions enabling the security of their systems and/or
 #  data to be ensured and,  more generally, to use and operate it in the
 #  same conditions as regards security.
 #
 #  The fact that you are presently reading this means that you have had
 #  knowledge of the CeCILL and CeCILL-C licenses and that you accept its terms.
 #
*/

// To compile this code :
//
// Windows : g++ -o decompress_clut decompress_clut.cpp -fopenmp -Ofast -lgdi32
// Linux : g++ -o decompress_clut decompress_clut.cpp -fopenmp -Ofast -lX11 -lpthread
//
// To run './decompress_clut [-i clut_name] [-r resolution]',
// where 'clut_name' is one of the CLUT name listed in 'gmic_cluts.txt'.
//
// Enabling OpenMP clearly speeds up the computation.

#include "./CImg.h"
using namespace cimg_library;

// Function used to compute gradient orientations of the distance function.
//-------------------------------------------------------------------------
float maxabs(const float a, const float b) {
  return std::abs(a)>std::abs(b)?a:b;
}

// Decompress a 3D CLUT from a number of input keypoints.
//
// - Size of input buffer 'keypoints' must be '6*nb_keypoints',
//    ordered as 'X0,Y0,Z0,R0,G0,B0,...,XN,YN,ZN,RN,GN,BN', for N+1 keypoints.
//
// - Size of output buffer 'output_clut_data' must be 'resolution^3*3*sizeof(float)'
//
//-----------------------------------------------------------------------------------
void decompress_clut(const unsigned char *const input_keypoints, const unsigned int nb_input_keypoints,
                     const unsigned int output_resolution, float *const output_clut_data) {
  CImg<float> img(1,1,1,3,0);  // We start with a 1x1x1 image (lowest resolution)
  while(true) {
    CImg<float> points(img.width(),img.height(),img.depth(),3,0);
    CImg<unsigned short> mask(img.width(),img.height(),img.depth(),1,0);

    // Prepare data at current scale.
    const unsigned char *p = input_keypoints;
    for (unsigned int n = 0; n<nb_input_keypoints; ++n) {
      const int
        X = (int)cimg::round(p[0]*(img.width() - 1.0f)/255),
        Y = (int)cimg::round(p[1]*(img.height() - 1.0f)/255),
        Z = (int)cimg::round(p[2]*(img.depth() - 1.0f)/255);
      points(X,Y,Z,0) += p[3];
      points(X,Y,Z,1) += p[4];
      points(X,Y,Z,2) += p[5];
      ++mask(X,Y,Z);
      p+=6;
    }

#pragma omp parallel for collapse(3)
    cimg_forXYZ(mask,x,y,z) if (mask(x,y,z)) {
      const unsigned short valm = mask(x,y,z);
      cimg_forC(points,c) points(x,y,z,c)/=valm;
      mask(x,y,z) = 1;
    }

    // Apply iterations of diffusion.
    if (mask.min()>0) points.move_to(img);
    else {
      CImg<float> eta, tmp(img);

      // The line below is the only 'serious' dependency to CImg, it returns the 3d distance function
      // to all keypoints at current resolution, computed in linear time (see CImg<T>::distance()).
      const CImg<float> dist = mask.get_distance(1);
      eta.assign(img.width(),img.height(),img.depth(),3);

#pragma omp parallel for collapse(3)
      cimg_forXYZ(dist,x,y,z) {
        const int
          px = x - 1, nx = x + 1,
          py = y - 1, ny = y + 1,
          pz = z - 1, nz = z + 1;
        const float i = dist(x,y,z);
        float ix,iy,iz;

        if (nx>=img.width()) ix = i - dist(px,y,z);
        else if (px<0) ix = dist(nx,y,z) - i;
        else ix = maxabs(i - dist(px,y,z), dist(nx,y,z) - i);

        if (ny>=img.height()) iy = i - dist(x,py,z);
        else if (py<0) iy = dist(x,ny,z) - i;
        else iy = maxabs(i - dist(x,py,z), dist(x,ny,z) - i);

        if (nz>=img.depth()) iz = i - dist(x,y,pz);
        else if (pz<0) iz = dist(x,y,nz) - i;
        else iz = maxabs(i - dist(x,y,pz), dist(x,y,nz) - i);

        const float norm = std::max(1e-5f,std::sqrt(ix*ix + iy*iy + iz*iz));
        eta(x,y,z,0) = ix/norm;
        eta(x,y,z,1) = iy/norm;
        eta(x,y,z,2) = iz/norm;
      }

      for (unsigned int i = 0; i<20; ++i) {

#pragma omp parallel for collapse(4)
        cimg_forXYZC(img,x,y,z,c) {
          if (mask(x,y,z)) tmp(x,y,z,c) = points(x,y,z,c);
          else {
            const float u = eta(x,y,z,0), v = eta(x,y,z,1), w = eta(x,y,z,2);
            tmp(x,y,z,c) = 0.5f*(img.cubic_atXYZ(x + u,y + v,z + w,c) + img.cubic_atXYZ(x - u,y - v,z - w,c));  // Use 3d Cubic interpolation
          }
        }
        img.swap(tmp);
      }
    }
    if (img.width()<output_resolution) {
      const unsigned int r = std::min(output_resolution,2*(unsigned int)img.width());
      img.resize(r,r,r,3,3); // Linear interpolation in 3D
    } else break;
  }
  img.cut(0,255); // Be sure all resulting colors are clamped in [0,255]
  std::memcpy(output_clut_data,img,img.size()*sizeof(float));
}

// Let's go folks!
//----------------
int main(int argc, char **argv) {

  // Get CLUT name from command line (must be one defined in 'gmic_cluts.txt').
  const char *const name = cimg_option("-i","summer","CLUT name");

  // Get CLUT resolution from command line, higher is slower.
  const unsigned int resolution = (unsigned int)cimg_option("-r",64,"CLUT resolution");

  // Get color image containing keypoint data for all available CLUTs.
  CImg<unsigned char> clut_data("gmic_cluts.ppm");

  // Get list of CLUT names.
  CImgList<char> clut_names = CImg<char>::get_load_raw("gmic_cluts.txt").get_split(CImg<char>("\n",1,1,1,1),0,false); //

  // Extract keypoint data for the specified CLUT name.
  const CImg<char> zero(1,1,1,1,0);
  unsigned int index = ~0U;
  cimglist_for(clut_names,l) if (!cimg::strcasecmp((clut_names[l],zero).get_append('y').data(),name)) { index = l; break; }
  if (index==~0U) throw CImgException("Specified CLUT name '%s' has not been found.",name);

  // Decompress requested CLUT from its set of keypoints.
  CImg<unsigned char> keypoints = clut_data.rows(2*index,2*index+1).get_split('y').get_append('c').permute_axes("cxyz");     // Reorganize keypoints data as a 6xN scalar image
  cimg_forY(keypoints,y) if (y && !keypoints(0,y) && !keypoints(1,y) && !keypoints(2,y)) { keypoints.rows(0,y - 1); break; } // Keep only significant keypoints
  float *const output_clut_data = new float[3*resolution*resolution*resolution]; // Output buffer for the CLUT data

  cimg::tic(); // Init tic/toc to display computation time
  decompress_clut(keypoints.data(),keypoints.size()/6,resolution,output_clut_data);
  cimg::toc();

  // Visualize CLUT as a 3D image.
  const CImg<float> clut3d = CImg<float>(output_clut_data,resolution,resolution,resolution,3,true);
  clut3d.display("3D CLUT");

  // Visualize CLUT as a 2D HaldCLUT.
  const unsigned int res2d = (unsigned int)std::floor(std::sqrt(resolution*resolution*resolution));
  const CImg<float> clut2d = CImg<float>(output_clut_data,res2d,res2d,1,3);
  clut2d.display("2D HaldCLUT");

  return 0;
}
