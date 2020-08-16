import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import os
import warnings
import mistree as mist
import random
import indexed_point
import indexed_edge

class training_data_generator:

    w = 100
    h = 100
    s = 10
    edge_index = 4
    startpoint = 0
    endpoint = 1
    branch_index = 5
    directory = "/home/70D4867/machinelearning/machinelearning/sequences"
    chance_of_point_placement = 40
    cluster_max_angle = 180
    cluster_radius = 250 # in decimeter
    min_distance_between_points = 20
    roundFactor = 2
    number_of_generations = 10

    def __init__(self, local_number_of_generations):
        self.number_of_generations = local_number_of_generations


    def isPointAlreadyUsed(self, point,pointArray):
        for usedPoint in pointArray:
            if usedPoint.x == point.x and usedPoint.y == point.y:
                return True
        return False

    def findEdgesContainingPoint(self, point,edgesToSort):
        # print('Start order with point -> x:'+str(point.x)+'y:'+str(point.y)+' and '+str(len(edgesToSort))+' possible edges')
        edgesWithPoint = []
        for edge in edgesToSort:
            if edge.startsOrEndsWithPoint(point):
                edgesWithPoint.append(edge)
        # print(' edgesWithPoint is returned with : '+str(len(edgesWithPoint)))
        return edgesWithPoint

    def orderEdgesFromRootPoint(self, point,edgesToSort):
        foundEdges = []
        for foundEdge in self.findEdgesContainingPoint(point,edgesToSort):
            if foundEdge.endPoint == point:
                foundEdge.switchOrientation()
            foundEdges.append(foundEdge)
        return foundEdges

    def treeAsEdgesFromRoot(self, point,edgesToSort):
        usedPoints=[]
        alledges = []
        foundedges = []
        foundedges = self.orderEdgesFromRootPoint(point,edgesToSort)
        usedPoints.append(point)
        for foundEdge in foundedges:
            alledges.append(foundEdge)
        iterations = len(edgesToSort)
        for index in range(iterations):
            edgesToIterate =[]
            for foundEdge in foundedges:
                point = foundEdge.endPoint
                if self.isPointAlreadyUsed(point, usedPoints):
                    continue
                newfoundedges=[]
                newfoundedges = self.orderEdgesFromRootPoint(point,edgesToSort)
                usedPoints.append(point)
                for newfoundedge in newfoundedges:
                    edgesToIterate.append(newfoundedge)
                    isdublicate = False
                    for oldedge in alledges:
                        if oldedge.edgeEqualsToAnotherEdge(newfoundedge):
                                isdublicate = True
                    if not isdublicate:
                        alledges.append(newfoundedge)
            if len(edgesToIterate) > 0:
                foundedges = []
                for edgeToIterate in edgesToIterate:
                    foundedges.append(edgeToIterate)
            else:
                for foundEdge in foundedges:
                    foundedges.remove(foundEdge)
                    newStartpoint = foundEdge.endPoint
                    newEndpoint = foundEdge.startPoint
                    foundEdge.startPoint = newStartpoint
                    foundEdge.endPoint = newEndpoint
                    edgesToIterate.append(foundEdge)
                foundedges = edgesToIterate
        return alledges


    def isFarEnough(self, pointOne, pointTwo):
        distance = math.hypot(pointTwo.x - pointOne.x, pointTwo.y - pointOne.y)
        if distance > self.min_distance_between_points:
            return True
        else:
            return False

    def generateCluster(self):
        cluster = []
        for x in range(0, self.cluster_max_angle):
            if random.randint(0,100) < self.chance_of_point_placement:
                radius = random.uniform(0.1, self.cluster_radius)
                angle = x
                possible_point = indexed_point.indexed_point(0,0.0,0.0,radius,angle)
                if not cluster:
                    cluster.append(possible_point)
                else:
                    farEnough = True
                    for ind_point in cluster:
                        if not self.isFarEnough(ind_point, possible_point):
                            farEnough = False
                    if farEnough:
                        cluster.append(possible_point)
        print('cluster generated with element count: '+ str(len(cluster)))
        return cluster

    def generateClusters(self, local_number_of_generations):
        cluster_array = []
        for g in range(1,local_number_of_generations):
            cluster_array.append(self.generateCluster())
        return cluster_array

    def getXandYCoordinatesOfClusterAsNumpyArrays(self, local_cluster):
        xCoords =[]
        yCoords =[]
        for idx_point in local_cluster:
            xCoords.append(idx_point.x)
            yCoords.append(idx_point.y)
        x = np.array(xCoords)
        y = np.array(yCoords)
        return x,y


    def printPointList(self, pointlist):
        for item in pointlist:
            print(item, end='\n')

    def printIndexedPointList(self, indexedPointlist):
        for index in range(len(indexedPointlist)):
            print(index,' : ',indexedPointlist[index].x,indexedPointlist[index].y)

    def fillCoordinateArrayAndIndexedPointarrayWithCoordinates(self, local_x,local_y):
        local_cluster_points= []
        local_hull_base = []
        for index in range(len(local_x)):
            point = indexed_point.indexed_point(index,local_x[index],local_y[index],0.0,0.0)
            local_cluster_points.append(point)
            local_hull_base.append([point.x,point.y])
        return local_cluster_points, local_hull_base

    def plotPointCluster(self, local_x,local_y):
        plt.figure(figsize=(8., 5.))
        plt.plot(local_x, local_y, "o")
        plt.show()

    def createHull(self, local_hull_base):
        local_points=np.array(local_hull_base)
        local_hull = ConvexHull(local_points)
        return local_hull, local_points

    def plotHullOfCluster(self, local_hull_base):
        print('HULL: \n')
        hull, local_points = self.createHull(local_hull_base)
        plt.plot(local_points[:,0], local_points[:,1], 'o')
        for simplex in hull.simplices:
            plt.plot(local_points[simplex, 0], local_points[simplex, 1], 'k')
        plt.figure(figsize=(8., 5.))
        plt.show()


    def plotMSTandSTATS(self, mst,local_x,local_y):
        plt.figure(figsize=(8., 5.))
        d, l, b, s, l_index, b_index = mst.get_stats(include_index=True)


        # plotting nodes:
        plt.scatter(local_x, local_y, s=10)

        # plotting branches:
        for i in range(0, len(b_index)):
            plt.plot([local_x[l_index[0][b_index[i][0]]], local_x[l_index[1][b_index[i][0]]]],
                    [local_y[l_index[0][b_index[i][0]]], local_y[l_index[1][b_index[i][0]]]],
                    color='C0', linestyle=':')
            plt.plot([local_x[l_index[0][b_index[i][1:-1]]], local_x[l_index[1][b_index[i][1:-1]]]],
                    [local_y[l_index[0][b_index[i][1:-1]]], local_y[l_index[1][b_index[i][1:-1]]]],
                    color='C0')
            plt.plot([local_x[l_index[0][b_index[i][-1]]], local_x[l_index[1][b_index[i][-1]]]],
                    [local_y[l_index[0][b_index[i][-1]]], local_y[l_index[1][b_index[i][-1]]]],
                    color='C0', linestyle=':')

        # ploting MST edges:
        plt.plot([local_x[l_index[0]], local_x[l_index[1]]],
                [local_y[l_index[0]], local_y[l_index[1]]],
                color='grey', linewidth=2, alpha=0.25)

        plt.plot([], [], color='C0', label=r'$Branch$ $Mid$')
        plt.plot([], [], color='C0', label=r'$Branch$ $End$', linestyle=':')
        plt.plot([], [], color='grey', alpha=0.25, label=r'$MST$ $Edges$')
        plt.xlabel(r'$X$', size=16)
        plt.ylabel(r'$Y$', size=16)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()

        # begins by binning the data and storing this in a dictionary.
        hmst = mist.HistMST()
        hmst.setup()
        mst_dict = hmst.get_hist(d, l, b, s)

        # plotting which takes as input the dictionary created before.
        pmst = mist.PlotHistMST()
        pmst.read_mst(mst_dict)
        pmst.plot()

    def createMSTandSTATS(self, mst,local_x,local_y):
        d, l, b, s, l_index, b_index = mst.get_stats(include_index=True)
        # begins by binning the data and storing this in a dictionary.
        hmst = mist.HistMST()
        hmst.setup()
        mst_dict = hmst.get_hist(d, l, b, s)

        # plotting which takes as input the dictionary created before.
        pmst = mist.PlotHistMST()
        pmst.read_mst(mst_dict)





    def defineRootIndexedPoints(self, local_hull_base):
        hull, local_points = self.createHull(local_hull_base)
        hull_indices = np.unique(hull.simplices.flat)
        hull_points = local_points[hull_indices, :]
        roots = []
        for item in hull_points:
            roots.append(indexed_point.indexed_point(0,item[0],item[1],0.0,0.0))
        return roots

    def buildMSTEdgesIndexedPointArray(self, mst, local_cluster_points):
        outArray = mst.output_stats(include_index=True)
        local_edges = []
        for index in range(len(outArray[self.edge_index][0])):
            local_edges.append(indexed_edge.indexed_edge(index, local_cluster_points[outArray[self.edge_index][self.startpoint][index]],local_cluster_points[outArray[self.edge_index][self.endpoint][index]]))
        return local_edges

    def printListOfEdgeVectors(self, local_edges):
            for index in range(len(local_edges)):
                sPosition = local_edges[index].startPoint
                ePosition = local_edges[index].endPoint
                print(index,' : ',sPosition.x,sPosition.y,sPosition.asPolarCoordinate(),ePosition.x,ePosition.y, ePosition.asPolarCoordinate())

    def sortEdgesFromHullpointRootsAndReturnAsArray(self, local_roots,local_edges):
        rootSortedEdges = []
        for hullpoint in local_roots:
            edgesFromThisRoot = []
            edgesToSort = local_edges
            edgesFromThisRoot = self.treeAsEdgesFromRoot(hullpoint,edgesToSort)
            if len(edgesFromThisRoot) == len(local_edges):
                rootSortedEdges.append(edgesFromThisRoot)
            orderedEdges = []
        return rootSortedEdges

    def consolidateAllSequencesToOneDataSetFromAllSortedEdgesOfGraph(self, local_rootSortedEdges):
        sequencedataset =[]
        for item in local_rootSortedEdges:
            #print('### sequence has '+str(len(item))+' elements ###')
            numericaledgeDataArray = []
            for index in range(len(item)):
                sPosition = item[index].startPoint
                ePosition = item[index].endPoint
                numericaledge = np.array([sPosition.x,sPosition.y,ePosition.x,ePosition.y])
                numericaledgeDataArray.append(numericaledge)
            sequencedataset.append(numericaledgeDataArray)
        return sequencedataset

    def produceTrainingSequences(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        datasetOfGraphSequences =[]
        clusters = self.generateClusters(self.number_of_generations)
        for point_cluster in clusters:
            x,y = self.getXandYCoordinatesOfClusterAsNumpyArrays(point_cluster)
            cluster_points, hull_base = self.fillCoordinateArrayAndIndexedPointarrayWithCoordinates(x,y)
            mst = mist.GetMST(x, y)
            roots = self.defineRootIndexedPoints(hull_base)
            self.createMSTandSTATS(mst,x,y)
            edges = self.buildMSTEdgesIndexedPointArray(mst,cluster_points)
            rootSortedEdges = self.sortEdgesFromHullpointRootsAndReturnAsArray(roots,edges)
            datasetOfGraphSequences.append(self.consolidateAllSequencesToOneDataSetFromAllSortedEdgesOfGraph(rootSortedEdges))
        return datasetOfGraphSequences

    def produceTrainingSequencesWithVisualLog(self):
        datasetOfGraphSequences =[]
        clusters = self.generateClusters(self.number_of_generations)
        for point_cluster in clusters:
            x,y = self.getXandYCoordinatesOfClusterAsNumpyArrays(point_cluster)
            print('\n Sample Start \n')
            print('CLUSTER: \n')
            cluster_points, hull_base = self.fillCoordinateArrayAndIndexedPointarrayWithCoordinates(x,y)
            self.printPointList(hull_base)
            self.plotPointCluster(x,y)
            self.plotHullOfCluster(hull_base)
            mst = mist.GetMST(x, y)
            self.plotMSTandSTATS(mst,x,y)
            print('\n Hull Nodes: \n')
            roots = self.defineRootIndexedPoints(hull_base)
            self.printIndexedPointList(roots)
            print('\n Tree Edges: \n')
            edges = self.buildMSTEdgesIndexedPointArray(mst,cluster_points)
            self.printListOfEdgeVectors(edges)
            rootSortedEdges = self.sortEdgesFromHullpointRootsAndReturnAsArray(roots,edges)
            print('\n Sorted Tree Edges: \n')
            datasetOfGraphSequences.append(self.consolidateAllSequencesToOneDataSetFromAllSortedEdgesOfGraph(rootSortedEdges))
            print('\n Sample End \n')
        return datasetOfGraphSequences
