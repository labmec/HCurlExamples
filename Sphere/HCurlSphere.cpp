/**********************************************************************
* This example aims to demonstrate the usage of HCurl-conforming      *
* elements using the NeoPZ library projecting a given solution in a   *
* Hcurl conforming mesh in two and three dimensions.                  *
**********************************************************************/

#include "pzgmesh.h"
#include "tpzcube.h"
#include "tpzgeoelrefpattern.h"
#include "TPZQuadSphere.h"

#include "TPZVTKGeoMesh.h"
#include "pzcheckgeom.h"

/// generate a cube with embedded sphere
TPZGeoMesh *SphereMesh(TPZVec<REAL> &center, REAL radius);


int main()
{
    TPZManVector<REAL,3> center(3,0.);
    REAL radius = 10.;
    TPZGeoMesh *gmesh = SphereMesh(center, radius);
    delete gmesh;
    return 0;
    
}


/// generate a cube with embedded sphere
TPZGeoMesh *SphereMesh(TPZVec<REAL> &center, REAL radius)
{
    TPZFNMatrix<24,REAL> refcoord(3,8,0.);
    for (int node=0; node<8; node++) {
        TPZManVector<REAL,3> co(3);
        pztopology::TPZCube::CenterPoint(node, co);
        for (int i=0; i<3; i++) {
            refcoord(i,node) = co[i];
        }
    }
    TPZGeoMesh *gmesh = new TPZGeoMesh();
    gmesh->SetDimension(3);
    gmesh->NodeVec().Resize(24);
    REAL dist = radius;
    for(int layer=0; layer<3; layer++)
    {
        for (int inode=0; inode<8; inode++)
        {
            TPZManVector<REAL,3> co(3,0.);
            for(int i=0; i<3; i++) co[i] = center[i]+refcoord(i,inode)*dist;
            gmesh->NodeVec()[8*layer+inode].Initialize(co, *gmesh);
        }
        dist /= 2.;
    }
    for(int layer=0; layer<2; layer++)
    {
        for(int face=0; face<6; face++)
        {
            int side = 20+face;
            TPZManVector<int64_t,8> nodeindices(8,-1);
            for(int in=0; in<4; in++)
            {
                nodeindices[in] = pztopology::TPZCube::SideNodeLocId(side, in)+8*layer;
                nodeindices[in+4] = nodeindices[in]+8;
            }
            int64_t index;
            gmesh->CreateGeoBlendElement(ECube, nodeindices, layer+1, index);
        }
    }
    {
        TPZManVector<int64_t,8> nodeindices(8,-1);
        for(int i=0; i<8; i++) nodeindices[i] = 16+i;
        int64_t index;
        gmesh->CreateGeoElement(ECube, nodeindices, 3, index);
    }
    int layer = 1;
    for(int face=0; face<6; face++)
    {
        int side = 20+face;
        TPZManVector<int64_t,8> nodeindices(4,-1);
        for(int in=0; in<4; in++)
        {
            nodeindices[in] = pztopology::TPZCube::SideNodeLocId(side, in)+8*layer;
        }
        int64_t index;
        TPZGeoElRefPattern<pzgeom::TPZQuadSphere<> > *gel =
        new TPZGeoElRefPattern<pzgeom::TPZQuadSphere<> >(nodeindices, 10,*gmesh);
        gel->Geom().SetData(radius*sqrt(3.)/2, center);
    }
    gmesh->BuildConnectivity();
    
    {
        TPZCheckGeom check(gmesh);
        check.UniformRefine(4);
    }
    {
        std::ofstream out("SphereInCube.vtk");
        TPZVTKGeoMesh::PrintGMeshVTK(gmesh, out);
    }
    return gmesh;
}

