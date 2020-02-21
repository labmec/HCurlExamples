/**********************************************************************
* This example aims to demonstrate the usage of HCurl-conforming      *
* elements using the NeoPZ library. It solves the Helmholtz equation  *
* in two and three dimensions using HCurl-conforming elements.        *
* The domain is a unit square/cube,  ranging from [0,1]^3 and         *
* dirichlet homogeneous conditions are imposed in all boundaries.     *
**********************************************************************/

#include "Analysis/pzanalysis.h"
#include "Material/TPZMatHelmholtz.h"
#include "Material/pzbndcond.h"
#include "Matrix/pzstepsolver.h"
#include "Mesh/pzgmesh.h"
#include "Mesh/TPZGeoMeshTools.h"
#include "Mesh/pzintel.h"
#include "Mesh/pzcondensedcompel.h"
#include "Mesh/TPZCompMeshTools.h"
#include "Post/TPZVTKGeoMesh.h"
#include "Shape/pzshapelinear.h"//in order to adjust the polynomial family to be used
#ifdef USING_MKL
#include "StrMatrix/TPZSSpStructMatrix.h"
#else
#include "StrMatrix/pzskylstrmatrix.h"
#endif


#include <string>

enum EOrthogonalFuncs{
    EChebyshev = 0,EExpo = 1,ELegendre = 2 ,EJacobi = 3,EHermite = 4
};

/**
* Generates a computational mesh that implements the problem to be solved
*/
static TPZCompMesh *CreateCompMesh(TPZGeoMesh *gmesh, const TPZVec<int> &matIds, const int initialPOrder, EOrthogonalFuncs familyType);

/**
 * This method will perform the P refinement on certain elements (adaptive P refinement).
 * These elements are the ones whose error exceed a given percentage of the maximum error in the mesh.
 */
static void PerformAdapativePRefinement(TPZCompMesh *cMesh, TPZAnalysis &an, const REAL errorPercentage);

/**
 * This method will perform uniform P refinement, i.e., all elements will have their polynomial order increased
 */
static void PerformUniformPRefinement(TPZCompMesh *cmesh, TPZAnalysis &an);

/**
 * This method is responsible to removing the equation corresponding to dirichlet boundary conditions. Used
 * if one desires to analyse the condition number of the resultant matrix
 */
static void FilterBoundaryEquations(TPZCompMesh *cmesh, TPZVec<int64_t> &activeEquations, int64_t &neq,
                             int64_t &neqOriginal);

void ExactSolution3D(const TPZVec<REAL> &pt, TPZVec<STATE> &sol, TPZFMatrix<STATE> &solDx){
    const auto & x = pt[0], y = pt[1], z = pt[2];
    sol.Resize(3);
    sol[0] = sin(2*M_PI*y)*sin(2*M_PI*z);
    sol[1] = sin(2*M_PI*x)*sin(2*M_PI*z);
    sol[2] = sin(2*M_PI*x)*sin(2*M_PI*y);
    solDx.Resize(3,1);
    solDx(0,0) =  2*M_PI*cos(2*M_PI*y)*sin(2*M_PI*x) - 2*M_PI*cos(2*M_PI*z)*sin(2*M_PI*x);
    solDx(1,0) = -2*M_PI*cos(2*M_PI*x)*sin(2*M_PI*y) + 2*M_PI*cos(2*M_PI*z)*sin(2*M_PI*y);
    solDx(2,0) =  2*M_PI*cos(2*M_PI*x)*sin(2*M_PI*z) - 2*M_PI*cos(2*M_PI*y)*sin(2*M_PI*z);
};

int main(int argc, char **argv)
{
#ifdef LOG4CXX
    InitializePZLOG();
#endif
    constexpr int numthreads{16};//number of threads to be used throughout the program
#ifdef USING_MKL
    mkl_set_dynamic(0); // disable automatic adjustment of the number of threads
    mkl_set_num_threads(numthreads);
#endif
    //physical dimension of the problem
    constexpr int dim{3};
    //number of divisions of each direction (x, y or x,y,z) of the domain
    constexpr int nDiv{6};
    //initial polynomial order
    constexpr int initialPOrder{1};
    //this will set how many rounds of p-refinements will be performed
    constexpr int nPRefinements{4};
    //this will set how many rounds of h-refinements will be performed
    constexpr int nHRefinements{0};
    //whether to calculate the errors
    constexpr bool calcErrors = true;
    //whether to perform adaptive or uniform p-refinement
    constexpr bool adaptiveP = false;
    //once the element with the maximum error is found, elements with errors bigger than
    //the following percentage will be refined as well
    constexpr REAL errorPercentage{0.3};
    //whether to apply static condensation on the internal dofs
    constexpr bool condense{false};
    //whether to remove the dirichlet boundary conditions from the matrix
    constexpr bool filterBoundaryEqs{true};
    //which family of polynomials to use
    EOrthogonalFuncs orthogonalPolyFamily = EChebyshev;//EChebyshev = 0,EExpo = 1,ELegendre = 2 ,EJacobi = 3,EHermite = 4
    //whether to generate .vtk files
    constexpr bool postProcess{false};
    constexpr int postProcessResolution{0};
    constexpr MMeshType meshType{MMeshType::ETetrahedral};
    constexpr bool exportConvergenceResults{true};

    if(!calcErrors && adaptiveP){
        std::cout<<"Either calculate the errors or choose uniform p-refinement. Aborting...\n";
        return -1;
    }
    const std::string executionInfo = [&](){
        std::string name("");
        switch(meshType){
            case MMeshType::ETriangular: name.append("_elTri");
            break;
            case MMeshType::EQuadrilateral: name.append("_elQuad");
            break;
            case MMeshType::ETetrahedral: name.append("_elTet");
            break;
            case MMeshType::EHexahedral: name.append("_elHex");
            break;
            case MMeshType::EHexaPyrMixed: name.append("_elHexPyr");
            break;
            case MMeshType::EPrismatic: name.append("_elPri");
            break;
        }
        if(adaptiveP) name.append("_adapP");
        else name.append("_unifP");
        name.append("_initialP");
        name.append(std::to_string(initialPOrder));
        name.append("_nPrefs");
        name.append(std::to_string(nPRefinements));
        name.append("_initialDiv");
        name.append(std::to_string(nDiv));
        name.append("_nHrefs");
        name.append(std::to_string(nHRefinements));
        return name;
    }();
    const std::string plotfile = "solution"+executionInfo+".vtk";//where to print the vtk files
    const std::string convfileName = "convRes"+executionInfo+".csv";//where to print the convergence results


    if(exportConvergenceResults){
        std::ofstream convfile;
        convfile.open(convfileName,std::ofstream::out | std::ofstream::trunc);//erases the content
        std::ostringstream fileInfo;
        fileInfo<<std::left<<std::setw(20)<<"h,"
                <<std::left<<std::setw(20)<<"p,"
                <<std::left<<std::setw(20)<<"nel,"
                <<std::left<<std::setw(20)<<"neq,"
                <<std::left<<std::setw(20)<<"e1,"
                <<std::left<<std::setw(20)<<"e2,"
                <<std::left<<std::setw(20)<<"e3";
        fileInfo<<"\n";
        convfile << fileInfo.str();
        convfile.close();
    }
    for(auto itH = 0 ; itH < nHRefinements + 1; itH++){
        std::cout<<"============================"<<std::endl;
        std::cout<<"\tIteration (h) "<<itH+1<<" out of "<<nHRefinements + 1<<std::endl;
        /** In NeoPZ, the TPZMaterial classes are used to implement the weak statement of the differential equation,
         * along with setting up the constitutive parameters of each region of the domain. See the method CreateCompMesh
         * in this file for an example.
         * The material ids are identifiers used in NeoPZ to identify different domain's regions/properties.
         */
        TPZVec<int> matIdVec(dim*2+1,2);
        matIdVec[0] = 1;//volumetric elements have a different identifier
        TPZGeoMesh *gMesh = [&]() -> TPZGeoMesh *{
            static TPZManVector<REAL,3> minX(3,0);
            static TPZManVector<REAL,3> maxX(3,1);
            TPZVec<int> nDivs(dim,nDiv * (itH + 1));
            return TPZGeoMeshTools::CreateGeoMeshOnGrid(dim,minX,maxX,matIdVec,nDivs,meshType,true);
        }();
        std::cout<<"\tNumber of elements: "<<gMesh->NElements()<<std::endl;
        //the HCurl functions on the pyramid are defined by imposing restriction over the functions of two tetrahedra
        if(meshType == MMeshType::EHexaPyrMixed || meshType == MMeshType::EPyramidal){
            TPZGeoMeshTools::DividePyramidsIntoTetra(gMesh);
        }
        //prints mesh
        {
            std::string geoMeshName("geoMesh"+executionInfo);
            if(nHRefinements) geoMeshName += "_hdiv_"+std::to_string(itH);
            if(postProcess){
                std::ofstream geoMeshVtk(geoMeshName+".vtk");
                TPZVTKGeoMesh::PrintGMeshVTK(gMesh, geoMeshVtk);
            }
            std::ofstream geoMeshTxt(geoMeshName+".txt");
            gMesh->Print(geoMeshTxt);
        }

        //creates computational mesh
        TPZCompMesh *cMesh = CreateCompMesh(gMesh,matIdVec,initialPOrder,orthogonalPolyFamily);
        //Setting up the analysis object
        constexpr bool optimizeBandwidth{true};
        TPZAnalysis an(cMesh, optimizeBandwidth); //Creates the object that will manage the analysis of the problem
        {
            //The TPZStructMatrix classes provide an interface between the linear algebra aspects of the library and
            //the Finite Element ones. Their specification (TPZSymetricSpStructMatrix, TPZSkylineStructMatrix, etc) are
            //also used to define the storage format for the matrices. In this example, the first one is the
            //CSR sparse matrix storage, and the second one the skyline, both in their symmetric versions.



            //I highly recommend running this program using the MKL libraries, the solving process will be
            //significantly faster.
#ifdef USING_MKL
            TPZSymetricSpStructMatrix matskl(cMesh);
#else
            TPZSkylineStructMatrix matskl(cMesh);
#endif
            matskl.SetNumThreads(numthreads);
            an.SetStructuralMatrix(matskl);
        }

        //setting solver to be used. ELDLt will default to Pardiso if USING_MKL is enabled.
        TPZStepSolver<STATE> step;
        step.SetDirect(ELDLt);
        an.SetSolver(step);
        if(calcErrors){
            //setting reference solution
            switch(dim){
                case 3:
                    an.SetExact(ExactSolution3D);
                    break;
                default:
                    DebugStop();
            }
            an.SetThreadsForError(numthreads);
        }

        //setting variables for post processing
        TPZStack<std::string> scalnames, vecnames;
        vecnames.Push("E");//print the state variable
        vecnames.Push("curlE");//print the curl of the state variable
        if(calcErrors)  scalnames.Push("Error");//print the error of each element
        if(adaptiveP)  scalnames.Push("POrder");//print the polynomial order of each element
        scalnames.Push("MaterialId");//print the material identifier of each element
        //resize the matrix that will store the error for each element
        cMesh->ElementSolution().Resize(cMesh->NElements(),3);
        TPZVec<int64_t> activeEquations;
        for(auto itP = 0 ; itP < nPRefinements + 1; itP++){

            {
                std::string compMeshName("compMesh"+executionInfo+"_steph_"+std::to_string(itH)+"_stepp_"+std::to_string(itP));
                std::ofstream compMeshTxt(compMeshName+".txt");
                cMesh->Print(compMeshTxt);
            }

            std::cout<<"\t============================"<<std::endl;
            std::cout<<"\t\tIteration (p) "<<itP+1<<" out of "<<nPRefinements + 1<<std::endl;
            if(condense) TPZCompMeshTools::CreatedCondensedElements(cMesh,false,false);
            if(condense || itP > 0 ) an.SetCompMesh(cMesh,optimizeBandwidth);
            if(filterBoundaryEqs){
                int64_t neqOriginal = -1, neqReduced = -1;
                activeEquations.Resize(0);
                FilterBoundaryEquations(cMesh,activeEquations, neqReduced, neqOriginal);
                an.StructMatrix()->EquationFilter().Reset();
                an.StructMatrix()->EquationFilter().SetNumEq(cMesh->NEquations());
                an.StructMatrix()->EquationFilter().SetActiveEquations(activeEquations);
            }else{
                an.StructMatrix()->EquationFilter().SetNumEq(cMesh->NEquations());
            }
            std::cout<<"\tAssembling matrix with NDoF = "<<an.StructMatrix()->EquationFilter().NActiveEquations()<<"."<<std::endl;
            an.Assemble(); //Assembles the global stiffness matrix (and load vector)
            std::cout<<"\tAssemble finished."<<std::endl;
            std::cout<<"\tSolving system..."<<std::endl;
            an.Solve();
            std::cout<<"\tSolving finished."<<std::endl;
            if(calcErrors){
                std::cout<<"\t\tCalculating errors..."<<std::endl;
                TPZVec<REAL> errorVec(3,0);
                an.PostProcessError(errorVec,true);
#ifndef PZDEBUG
                std::cout << "############" << std::endl;
                std::cout <<"HCurl Norm   -> u = "  << sqrt(errorVec[0]) << std::endl;
                std::cout <<"L2 Norm      -> u = "  << sqrt(errorVec[1]) << std::endl;
                std::cout <<"L2 Norm- > curl U = "  << sqrt(errorVec[2])  <<std::endl;
                std::cout<<"############"<<std::endl;
#endif
                if(exportConvergenceResults){
                    std::ostringstream res;
                    typedef std::numeric_limits< double > dbl;
                    res.precision(dbl::max_digits10);
                    //"h    p    nel    neq    e1    e2    e3\n";
                    res << std::left << std::setw(20)<< std::fixed << 1./(nDiv * (itH + 1)) <<","
                        << std::left << std::setw(20)<< std::fixed << initialPOrder + itP <<","
                        << std::left << std::setw(20)<< std::fixed << cMesh->NElements() <<","
                        << std::left << std::setw(20)<< std::fixed << an.StructMatrix()->EquationFilter().NActiveEquations() <<","
                        << std::left << std::setw(20)<< std::fixed << errorVec[0] <<","
                        << std::left << std::setw(20)<< std::fixed << errorVec[1] <<","
                        << std::left << std::setw(20)<< std::fixed << errorVec[2] <<"\n";
                    std::ofstream convfile;
                    convfile.open(convfileName,std::ofstream::out | std::ofstream::app);
                    convfile<<res.str();
                    convfile.close();
                }
            }
            if(postProcess){
                std::cout<<"\t\tPost processing..."<<std::endl;
                an.DefineGraphMesh(dim, scalnames, vecnames, plotfile);
                an.SetStep(itH*(nPRefinements+1) + itP);
                an.PostProcess(postProcessResolution);
                std::cout<<"\t\tPost processing finished."<<std::endl;
            }
            if(itP < nPRefinements){
                if(adaptiveP)   PerformAdapativePRefinement(cMesh, an, errorPercentage);
                else PerformUniformPRefinement(cMesh,an);
            }
            if(condense) TPZCompMeshTools::UnCondensedElements(cMesh);
        }
        delete cMesh;
        delete gMesh;
    }
    return 0;
}

TPZCompMesh *CreateCompMesh(TPZGeoMesh *gmesh, const TPZVec<int> &matIds, const int initialPOrder, EOrthogonalFuncs familyType){
    TPZCompMesh *cmesh = new TPZCompMesh(gmesh);

    //Definition of the approximation space
    const int dim = gmesh->Dimension();
    cmesh->SetDefaultOrder(initialPOrder);
    cmesh->SetDimModel(dim);


    const int matId = matIds[0];
    constexpr REAL c{1};
    //Inserting material
    auto mat = new TPZMatHelmholtz(dim,matId,c,1);


    auto forcingFunction3D = [](const TPZVec<REAL>& pt, TPZVec<STATE> &result){
        static const auto aux = 8*M_PI*M_PI;

        const REAL &x = pt[0];
        const REAL &y = pt[1];
        const REAL &z = pt[2];
        result[0] = sin(2*M_PI*y)*sin(2*M_PI*z) + aux*sin(2*M_PI*y)*sin(2*M_PI*z);
        result[1] = sin(2*M_PI*x)*sin(2*M_PI*z) + aux*sin(2*M_PI*x)*sin(2*M_PI*z);
        result[2] = sin(2*M_PI*x)*sin(2*M_PI*y) + aux*sin(2*M_PI*x)*sin(2*M_PI*y);
    };

    constexpr int pOrderForcingFunction{10};
    switch(dim){
        case 3:
            mat->SetForcingFunction(forcingFunction3D,pOrderForcingFunction);
            break;
        default:
            DebugStop();
    }

    //Inserting volumetric materials objects
    cmesh->InsertMaterialObject(mat);

    //Boundary conditions
    constexpr int dirichlet{0};
    constexpr int neumann{1};

    const int &matIdBc1 = matIds[1];
    auto bc1 = mat->CreateBC(mat, matIdBc1, dirichlet, TPZFNMatrix<1,STATE>(1,1,0),TPZFNMatrix<1,STATE>(1,1,0));
    cmesh->InsertMaterialObject(bc1);

    cmesh->SetAllCreateFunctionsHCurl();//set Hcurl approximation space
    cmesh->AutoBuild();
    cmesh->AdjustBoundaryElements();
    cmesh->CleanUpUnconnectedNodes();

    switch(familyType){
        case EChebyshev:
            pzshape::TPZShapeLinear::fOrthogonal =  pzshape::TPZShapeLinear::Chebyshev;
            break;
        case EExpo:
            pzshape::TPZShapeLinear::fOrthogonal =  pzshape::TPZShapeLinear::Expo;
            break;
        case ELegendre:
            pzshape::TPZShapeLinear::fOrthogonal =  pzshape::TPZShapeLinear::Legendre;
            break;
        case EJacobi:
            pzshape::TPZShapeLinear::fOrthogonal =  pzshape::TPZShapeLinear::Jacobi;
            break;
        case EHermite:
            pzshape::TPZShapeLinear::fOrthogonal =  pzshape::TPZShapeLinear::Hermite;
            break;
    }

    return cmesh;
}

void PerformAdapativePRefinement(TPZCompMesh *cmesh, TPZAnalysis &an,
                                 const REAL errorPercentage) {
    std::cout<<"\tPerforming adaptive p-refinement..."<<std::endl;
    const auto nElems = cmesh->Reference()->NElements();
    // Iterates through element errors to get the maximum value
    REAL maxError = -1;
    for (int64_t iel = 0; iel < nElems; iel++) {
        TPZCompEl *cel = cmesh->ElementVec()[iel];
        if (!cel) continue;
        if (cel->Dimension() != cmesh->Dimension()) continue;
        REAL elementError = cmesh->ElementSolution()(iel, 0);
        if (elementError > maxError) {
            maxError = elementError;
        }
    }
    std::cout<<"\tMax error found (in one element): "<<maxError<<std::endl;
    // Refines elements which errors are bigger than 30% of the maximum error
    const REAL threshold = errorPercentage * maxError;
    int count = 0;
    for (int64_t iel = 0; iel < nElems; iel++) {
        auto cel =  [&](){
            auto cel1 = dynamic_cast<TPZInterpolationSpace *> (cmesh->Element(iel));
            if(cel1) return cel1;
            auto cel2 = dynamic_cast<TPZCondensedCompEl*> (cmesh->Element(iel));
            if(!cel2) return (TPZInterpolationSpace *)nullptr;
            auto cel3 = dynamic_cast<TPZInterpolationSpace *> (cel2->ReferenceCompEl());
            return cel3;
        }();
        if (!cel || cel->Dimension() != cmesh->Dimension()) continue;
        auto celIndex = cel->Index();
        REAL elementError = cmesh->ElementSolution()(celIndex, 0);
        if (elementError > threshold) {
            const int currentPorder = cel->GetPreferredOrder();
            cel->PRefine(currentPorder+1);
            count++;
        }
    }
    std::cout<<"\t"<<count<<" elements were refined in this step."<<std::endl;
    cmesh->AdjustBoundaryElements();
    cmesh->CleanUpUnconnectedNodes();
    cmesh->ExpandSolution();
}

void PerformUniformPRefinement(TPZCompMesh *cmesh, TPZAnalysis &an) {
    std::cout<<"\tPerforming uniform p-refinement..."<<std::endl;
    const auto nElems = cmesh->Reference()->NElements();
    int count = 0;
    for (int64_t iel = 0; iel < nElems; iel++) {
        auto cel =  [&](){
            auto cel1 = dynamic_cast<TPZInterpolationSpace *> (cmesh->Element(iel));
            if(cel1) return cel1;
            auto cel2 = dynamic_cast<TPZCondensedCompEl*> (cmesh->Element(iel));
            if(!cel2) return (TPZInterpolationSpace *)nullptr;
            auto cel3 = dynamic_cast<TPZInterpolationSpace *> (cel2->ReferenceCompEl());
            return cel3;
        }();
        if (!cel || cel->Dimension() != cmesh->Dimension()) continue;
        const int currentPorder = cel->GetPreferredOrder();
        cel->PRefine(currentPorder+1);
        count++;
    }
    std::cout<<"\t"<<count<<" elements were refined in this step."<<std::endl;
    cmesh->AdjustBoundaryElements();
    cmesh->CleanUpUnconnectedNodes();
    cmesh->ExpandSolution();
}

void FilterBoundaryEquations(TPZCompMesh *cmesh, TPZVec<int64_t> &activeEquations, int64_t &neq,
                             int64_t &neqOriginal) {
    neqOriginal = cmesh->NEquations();
    neq = 0;

    std::cout << "Filtering boundary equations..." << std::endl;
    TPZManVector<int64_t, 1000> allConnects;
    std::set<int64_t> boundConnects;

    for (auto iel = 0; iel < cmesh->NElements(); iel++) {
        TPZCompEl *cel = cmesh->ElementVec()[iel];
        if (cel == nullptr || cel->Reference() == nullptr) {
            continue;
        }
        TPZBndCond *mat = dynamic_cast<TPZBndCond *>(cmesh->MaterialVec()[cel->Reference()->MaterialId()]);

        //dirichlet boundary condition
        if (mat && mat->Type() == 0) {
            std::set<int64_t> boundConnectsEl;
            cel->BuildConnectList(boundConnectsEl);

            for (auto val : boundConnectsEl) {
                if (boundConnects.find(val) == boundConnects.end()) {
                    boundConnects.insert(val);
                }
            }
        }
    }

    for (auto iCon = 0; iCon < cmesh->NConnects(); iCon++) {
        if (boundConnects.find(iCon) == boundConnects.end()) {
            TPZConnect &con = cmesh->ConnectVec()[iCon];
            if(con.IsCondensed() || con.HasDependency()) continue;
            const int seqnum = con.SequenceNumber();
            if(seqnum < 0) continue;
            const int pos = cmesh->Block().Position(seqnum);
            const int blocksize = cmesh->Block().Size(seqnum);
            if (blocksize == 0) continue;
            const int vs = activeEquations.size();
            activeEquations.Resize(vs + blocksize);
            for (int ieq = 0; ieq < blocksize; ieq++) {
                activeEquations[vs + ieq] = pos + ieq;
                neq++;
            }
        }
    }
    std::cout << "# equations(before): " << neqOriginal << std::endl;
    std::cout << "# equations(after): " << neq << std::endl;
}