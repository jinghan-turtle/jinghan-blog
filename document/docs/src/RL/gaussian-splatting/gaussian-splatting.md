# 3D Gaussian Splatting

3DGS[<sup>[1]</sup>](#3DGS-paper) 是基于 Splatting 和机器学习的三维重建方法。其中 Splat 是拟声词，意为“啪叽一声”：我们可以想象三维场景重建的输入是一些雪球，图片是一面砖墙，图像生成过程就是向墙面扔雪球的过程；每扔一个雪球，墙面上会留有印记，并伴有啪叽一声，所以这个算法也被称为抛雪球法，翻译成“喷溅”也很有灵性。简单来说，splatting 的核心有三步：一是选择“雪球”，也就是说我要将它捏成一个什么形状的雪球；二是去抛掷雪球，将高斯椭球从 3D 投影到 2D，得到很多个印记；三是合成这些印记以形成最后的图像[<sup>[2]</sup>](#refer-anchor-2)。

## 捏雪球：用协方差控制椭球形状

3DGS 的输入是 SfM 得到的稀疏点云，而由于点是没有体积的，我们首先需要将点膨胀成正方体、球体或者其他基础的三维几何形状。之所以选择高斯分布作为椭球，则是因为它良好的数学性质，比如高斯分布在仿射变换后依然是高斯分布，而沿着某个轴积分将高斯分布从 3D 降到 2D 后其依然服从高斯分布。高斯分布的数学描述如下：

$$
\small
G(x;\mu,\Sigma) = \cfrac{1}{\sqrt{(2\pi)^k|\Sigma|}}\exp\bigg(-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)\bigg) 
% = \cfrac{1}{\sqrt{(2\pi)^3|\Sigma|}}\exp\Bigg(-\frac{1}{2}\bigg(\frac{(x-\mu_1)^2}{\sigma_1^2}+\frac{(y-\mu_2)^2}{\sigma_2^2}+\frac{(z-\mu_3)^2}{\sigma_3^2}-\frac{2\sigma_{xy}(x-\mu_1)(y-\mu_2)}{\sigma_1\sigma_2}-\frac{2\sigma_{xz}(x-\mu_1)(z-\mu_3)}{\sigma_1\sigma_3}-\frac{2\sigma_{yz}(y-\mu_2)(z-\mu_3)}{\sigma_2\sigma_3}\bigg)\Bigg)
$$

同时，任意椭球都可以由某个椭球经过仿射变换得到（这其实对应于从世界坐标系到相机坐标系的观测变换，所谓“横看成岭侧成峰，远近高低各不同”，在这里指的就是不同视角下看到的椭球形状是不同的），而仿射变换左乘的矩阵 $\small W$ 可以视为旋转和缩放这两个作用的合成，即 $\small W=RS$：

$$
\small
y = Wx+b = RSx+b,\thinspace\thinspace x\sim N(\mu,\Sigma) \thinspace\thinspace\thinspace\thinspace\thinspace\Longrightarrow\thinspace\thinspace\thinspace\thinspace\thinspace y\sim N(W\mu+b, W\Sigma W^T) = N(W\mu+b, RS\Sigma S^TR^T)
$$

特别地，当 $\small x$ 服从标准正态分布时，仿射变换得到的协方差矩阵为 $\small RSS^TR^T$；反过来，给定协方差矩阵 $\small\Sigma$，我们可以通过特征分解得到 $\small R$ 和 $\small S$，即 $\small\Sigma=Q\wedge Q^T=Q\wedge^{1/2}\wedge^{1/2} Q^T:=RSS^TR^T$ (由此可知存储一个协方差矩阵需要七个参数，即四元数和三个缩放参数)。下面的 `computeCov3D` 函数就在讲这个仿射变换，传入的三维向量 `scale` 即为上述公式中的 $\small x$， `cov3D` 则表示协方差矩阵，只是传入的四元数 `rot4` 使得代码多了一个计算旋转矩阵的过程。

//// collapse-code
```C++ hl_lines="26-29"
/* submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu */
// Forward method for converting scale and rotation properties of each Gaussian to 
// a 3D covariance matrix in world space. Also takes care of quaternion normalization.
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
    // Create scaling matrix
    glm::mat3 S = glm::mat3(1.0f);
    S[0][0] = mod * scale.x;
    S[1][1] = mod * scale.y;
    S[2][2] = mod * scale.z;

    // Normalize quaternion to get valid rotation
    glm::vec4 q = rot;
    float r = q.x;
    float x = q.y;
    float y = q.z;
    float z = q.w;

    // Compute rotation matrix from quaternion
    glm::mat3 R = glm::mat3(
        1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
        2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
        2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
    );

    glm::mat3 M = S * R;

    // Compute 3D world covariance matrix Sigma
    glm::mat3 Sigma = glm::transpose(M) * M;

    // Covariance is symmetric, only store upper right
    cov3D[0] = Sigma[0][0];
    cov3D[1] = Sigma[0][1];
    cov3D[2] = Sigma[0][2];
    cov3D[3] = Sigma[1][1];
    cov3D[4] = Sigma[1][2];
    cov3D[5] = Sigma[2][2];
}
```
////

## 抛雪球：将三维椭球投影到二维

从世界坐标系到相机坐标系的观测变换通过上面的仿射变换描述，而相机坐标系到归一化坐标系的投影变换却并不是一个线性变换，它需要将视锥的屁股压扁并压成正方体（这样一来也将射线与坐标轴平行对齐，使得沿射线的积分计算变得更加方便），所以我们考虑引入雅可比矩阵对该非线性变换作局部线性近似，也就是用仿射变换来近似局部的投影作用。那么，在该压扁的射线坐标系下的协方差矩阵为 $\small\Sigma_{ray}=JW\Sigma W^TJ^T$，而均值本身便是高斯椭球的中心点，可直接对其应用投影变换。如此，再对投影后的高斯椭圆作视口变换便可得到其在像素坐标系下的表示。

<!-- $$
\small
\frac{1}{z}
\begin{bmatrix}
    n & 0 & 0 & 0 \\
    0 & n & 0 & 0 \\
    0 & 0 & n+f & -nf \\
    0 & 0 & 1 & 0
\end{bmatrix}
\begin{bmatrix}
    x \\ y \\ z \\ 1
\end{bmatrix}
=
\begin{bmatrix}
    nx/z \\ ny/z \\ n+f-nf/z \\ 1
\end{bmatrix}
\thinspace\thinspace\thinspace
\Longrightarrow
\thinspace\thinspace\thinspace
J = 
\begin{bmatrix}
    n/z & 0 & -nx/z^2 \\
    0 & y/z & -ny/z^2 \\
    0 & 0 & -nf/z^2
\end{bmatrix}
$$ -->

![](./perspective-projection-with-formula.png){ width=90% style="display: block; margin: 0 auto;" }

正因为是局部线性近似，所以下面投影变换的 `computeCov2D` 函数需要先计算高斯椭球均值点在视锥中的位置；也正因为视锥压扁后的正交投影与 $\small z$ 方向无关，所以实际上雅可比矩阵的第三行是可以置零的。

//// collapse-code
```C++ hl_lines="17-20"
/* submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu */
// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
{
	// The following models the steps outlined by equations 29 and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	float3 t = transformPoint4x3(mean, viewmatrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}
```
////

## 雪球颜色和像素着色：球谐函数

通过上述过程，我们已经捏好了雪球，也想好如何把雪球往墙上砸了，但雪球不一定是白色的 —— 3DGS 利用球谐函数 $\small\sum_l\sum_{m=-l}^l c_l^my_l^m(\theta,\phi)$ 来表达高斯椭球的颜色。相比 RGB 信息（对应于零阶球谐函数），高阶球谐函数给出了更为逼真的环境贴图和亮度重建效果，使得椭球呈现的颜色与观测方向相关 —— 直觉上讲球谐函数包含了更为丰富的信息，比如三阶球谐函数所包含的信息量达到了 $\small (1+3+5+7)\times 3$。下面代码传入的参数 `deg` 即为球谐函数的阶数，`glm::vec3 result = SH_C0 * sh[0]` 便是在算第零阶的元素，后续则按公式分别计算不同阶次的球谐函数值。

//// collapse-code
``` C++
/* submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu */
// Forward method for converting the input spherical harmonics coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}
```
////

$\small\alpha-blending$ 中的像素颜色是通过沿射线的体渲染得到的，即将高斯椭球按照射线坐标系的深度排序，然后按从近到远的顺序依次抛出：$\small C=\sum_{i=1}^N T_i\alpha_ic_i=\sum_{i=1}^N\prod_{j=1}^{i-1}(1-\alpha_i)\big(1-\exp(-\sigma_i\delta_i)\big)c_i$，其中 $\small T(s)$ 表示在 $\small s$ 点之前光线没有被阻碍的概率或者说透过率，$\small\sigma(s)$ 表示在 $\small s$ 点处光线撞击粒子或者说被粒子阻碍的概率，$\small c(s)$ 表示在 $\small s$ 点处粒子发出的颜色，$\small\delta(s)$ 则表示点 $\small s$ 处沿射线离散积分的间距。下面代码的高亮部分对应的就是上述公式。

//// collapse-code
``` C++ hl_lines="86-102"
/* submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu */
// Main rasterization method. Collaboratively works on one tile per block, 
// each thread treats one pixel. Alternates between fetching and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	const float* __restrict__ depths,
	float* __restrict__ invdepth)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };

	float expected_invdepth = 0.0f;

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			float alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f)
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;

			if(invdepth)
			expected_invdepth += (1 / depths[collected_id[j]]) * alpha * T;

			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];

		if (invdepth)
		invdepth[pix_id] = expected_invdepth;// 1. / (expected_depth + T * 1e3);
	}
}
```
////

## 完整流程：机器学习与参数评估

每个点膨胀成的三维高斯椭球参数包括中心点位置 $\small (x,y,z)$、协方差矩阵 $\small\Sigma=RS$、球谐函数系数矩阵和透明度 $\small\alpha$，这些初始化的高斯椭球通过上述泼溅的过程得到二维图像，再将该图像和 Ground Truth 的误差反向传播来优化椭球参数，其中损失函数被定义为 $\small\mathcal{L}=(1-\lambda)\mathcal{L}_1 + \lambda\mathcal{L}_{D-SSIM}$，如下述代码块所示（可以注意到代码中还计算了深度正则化损失来引导高斯椭球的几何分布与单目先验深度估计保持一致，而采用逆深度图则是因为近处的深度估计更为准确）。可以注意到的是，代码 `submodules` 模块下有 `simple-knn` 部分，这是因为高斯椭球被初始化为一个各向同性的球，其半径被设为三近邻距离的平均值以避免椭球铺不满场景或者过度重叠的情况。

//// collapse-code
``` Python hl_lines="12"
''' train.py '''
def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
	...
	# Loss
	gt_image = viewpoint_cam.original_image.cuda()
	Ll1 = l1_loss(image, gt_image)
	if FUSED_SSIM_AVAILABLE:
		ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
	else:
		ssim_value = ssim(image, gt_image)

	loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

	# Depth regularization
	Ll1depth_pure = 0.0
	if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
		invDepth = render_pkg["depth"]
		mono_invdepth = viewpoint_cam.invdepthmap.cuda()
		depth_mask = viewpoint_cam.depth_mask.cuda()

		Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
		Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure
		loss += Ll1depth
		Ll1depth = Ll1depth.item()
	else:
		Ll1depth = 0

	loss.backward()
```
////

但是如果只对 colmap 生成的初始点云作优化的话，那么后续不管如何优化高斯椭球的数量都是不变的，这使得算法强依赖于 SfM 的初始化，所以 3DGS 提出了自适应密度控制与优化，即对透明的高斯分布作周期性滤除或者说剔除存在感太低的高斯椭球，同时，对于 under-reconstruction 的区域，克隆高斯并沿着梯度方向移动以覆盖几何体；对于 over-reconstruction 的区域则拆分高斯以更好地拟合细粒度细节。可以发现，机器学习的部分非常简单且不涉及深度学习的知识，3DGS 的难度主要在于对计算机图形学的理解和 GPU 的高性能编程。3DGS 这部分的伪代码见论文的附录 B. Optimization and Densification Algorithm。

![](./overview_of_3DGS.png){ width=100% style="display: block; margin: 0 auto;" }

<!-- ## 代码运行： Ubuntu 20.04 & Cuda 11.8

```
conda env create --file environment.yml && conda activate gaussian_splatting
```

运行 `python train.py --help` 指令可以获得其支持的命令行参数，从终端输出 `--source_path SOURCE_PATH, -s SOURCE_PATH` 可以看出 `-s` 对应的是源数据路径。

```
python train.py -s data/truck/ -m data/truck/output
``` -->

注：网上有反映说原版 3DGS 内置的查看器不太好用，可以考虑换用 [gaussian-splatting-lightning](https://github.com/yzslab/gaussian-splatting-lightning)。

&nbsp;

<div id="3DGS-paper"></div>
[1] [Kerbl B, Kopanas G, Leimkühler T, et al. 3D Gaussian splatting for real-time radiance field rendering[J]. ACM Trans. Graph., 2023, 42(4): 139:1-139:14.](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)

<div id="refer-anchor-2"></div>
[2] [Bilibili 上的这个视频给出了 3DGS 人性化的讲解，也是我这篇笔记的来源。](https://www.bilibili.com/video/BV1zi421v7Dr/?spm_id_from=333.337.search-card.all.click&vd_source=df8c598e3026e471e571a5970603f409)