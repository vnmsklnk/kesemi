import g2o
import numpy as np


class PoseOptimizer:
    def __init__(self, camera_intrinsics, reproj_tresh):
        self.reproj_tresh = reproj_tresh
        self.N_iter = 4
        self.min_N_edges = 20
        self.fx = camera_intrinsics[0, 0]
        self.fy = camera_intrinsics[1, 1]
        self.cx = camera_intrinsics[0, 2]
        self.cy = camera_intrinsics[1, 2]

    def optimize(self, kpts_3d_map, kpts_obs):
        edges = []

        optimizer = g2o.SparseOptimizer()
        block_solver = g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(block_solver)
        optimizer.set_algorithm(solver)

        v1 = g2o.VertexSE3Expmap()
        v1.set_id(0)
        v1.set_fixed(False)
        optimizer.add_vertex(v1)

        v1.set_estimate(g2o.SE3Quat(np.eye(3), np.zeros((3,))))

        for i in range(len(kpts_obs)):
            edge = g2o.EdgeSE3ProjectXYZOnlyPose()

            edge.set_vertex(0, v1)
            edge.set_measurement(kpts_obs[i][:2])
            edge.set_information(np.eye(2))
            edge.set_robust_kernel(g2o.RobustKernelHuber())

            edge.fx = self.fx
            edge.fy = self.fy
            edge.cx = self.cx
            edge.cy = self.cy
            edge.Xw = kpts_3d_map[i]

            optimizer.add_edge(edge)
            edges.append(edge)

        inl_mask = np.ones((len(kpts_obs),), dtype=np.bool)

        for i in range(self.N_iter):
            v1.set_estimate(g2o.SE3Quat(np.eye(3), np.zeros((3,))))

            optimizer.initialize_optimization()
            optimizer.optimize(10)

            for j, edge in enumerate(edges):
                if edge.chi2() > self.reproj_tresh / (i + 1) ** 2:
                    inl_mask[j] = False
                    edge.set_level(1)
                else:
                    inl_mask[j] = True
                    edge.set_level(0)

                if i == self.N_iter - 2:
                    edge.set_robust_kernel(None)

            if np.sum(inl_mask) < self.min_N_edges:
                break

        return v1.estimate().matrix()