/**
 * @file alpha_blending_enhanced.cu
 * @brief
 */

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <glm/glm.hpp>
#include <torch/extension.h>
#include <torch/torch.h>
#include <utils.h>
#include <vector>
#include <math.h>

// 添加Gabor相关常量
#define TOTAL_NUM_FREQUENCIES 16
#define SELECTED_NUM_FREQUENCIES 2
const float max_frequency = 10.0f;  // 可根据实际需要调整

namespace cg = cooperative_groups;

template <uint32_t CNum>
__global__ void __launch_bounds__(BLOCK_X *BLOCK_Y)
    renderGaussianEllipseForwardCUDAKernel(const int P,
                                   const float2 *__restrict__ uv,
                                   const float3 *__restrict__ conic,
                                   const float *__restrict__ opacity,
                                   const float *__restrict__ feature,
                                   const int *__restrict__ idx_sorted,
                                   const int2 *__restrict__ tile_range,
                                   const float bg,
                                   const int C,
                                   const int W,
                                   const int H,
                                   const int K,
                                   const bool enable_truncation,
                                   float *__restrict__ final_T,
                                   int *__restrict__ ncontrib,
                                   int *__restrict__ final_idx,
                                   float *__restrict__ rendered_feature) {
    auto block = cg::this_thread_block();
    int32_t tile_grid_x = (W + BLOCK_X - 1) / BLOCK_X;
    int32_t tile_id =
        block.group_index().y * tile_grid_x + block.group_index().x;
    uint2 pix = {block.group_index().x * BLOCK_X + block.thread_index().x,
                 block.group_index().y * BLOCK_Y + block.thread_index().y};
    uint32_t pix_id = W * pix.y + pix.x;
    float2 pixf = {(float)pix.x, (float)pix.y};
    const int c_num = min(CNum, C);

    bool inside = pix.x < W && pix.y < H;
    bool done = !inside;

    int2 range = tile_range[tile_id];
    const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
    int toDo = range.y - range.x;

    __shared__ int collected_id[BLOCK_SIZE];
    __shared__ float2 collected_uv[BLOCK_SIZE];
    __shared__ float3 collected_conic[BLOCK_SIZE];
    __shared__ float collected_opacity[BLOCK_SIZE];

    float T = 1.0f;
    uint32_t contributor = 0;
    uint32_t last_contributor = 0;
    float F[CNum] = {0};
    int layer_cnt = 0;

    for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE) {
        int num_done = __syncthreads_count(done);
        if (num_done == BLOCK_SIZE)
            break;

        int progress = i * BLOCK_SIZE + block.thread_rank();
        if (range.x + progress < range.y) {
            int coll_id = idx_sorted[range.x + progress];
            collected_id[block.thread_rank()] = coll_id;
            collected_uv[block.thread_rank()] = uv[coll_id];
            collected_conic[block.thread_rank()] = conic[coll_id];
            collected_opacity[block.thread_rank()] = opacity[coll_id];
        }
        block.sync();

        for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++) {
            contributor++;
            float2 vec = {collected_uv[j].x - pixf.x,
                          collected_uv[j].y - pixf.y};
            
            // Gabor 核计算
            float2 dx = vec;
            float theta = 0.5f; // 方向
            float cosr = cos(theta);
            float sinr = sin(theta);
            
            // 转换坐标到旋转空间
            float x_theta = dx.x * cosr + dx.y * sinr;
            float y_theta = -dx.x * sinr + dx.y * cosr;
            
            // 高斯部分计算
            float gaussian_part = -0.5f * (collected_conic[j].x * vec.x * vec.x +
                                   collected_conic[j].z * vec.y * vec.y) -
                          collected_conic[j].y * vec.x * vec.y;
            
            // 跳过高斯部分较弱的点
            if (gaussian_part > 0.0f)
                continue;
            
            float g_factor = exp(gaussian_part);
            
            // 使用与提供代码相同的Gabor条纹计算
            float sinusoid_part = 0.0f;
            for(int w_idx = 0; w_idx < SELECTED_NUM_FREQUENCIES; w_idx++){
                float f = max_frequency * (w_idx + 1) / (float)TOTAL_NUM_FREQUENCIES;
                sinusoid_part += 0.5f + 0.5f * cos(f * x_theta);
            }
            sinusoid_part /= SELECTED_NUM_FREQUENCIES; // 归一化
            
            // 直接使用计算的sinusoid_part作为调制因子
            float alpha = min(0.99f, collected_opacity[j] * g_factor * sinusoid_part);
            if (alpha < 1.0 / 255.0)
                continue;

            float next_T = T * (1 - alpha);
            if (next_T < 0.0001f) {
                done = true;
                continue;
            }

            for (int k = 0; k < c_num; k++)
                F[k] += feature[k * P + collected_id[j]];

            T = 0.0f;
            done = true;
            break;

            T = next_T;
            last_contributor = contributor;

            if (enable_truncation) {
                final_idx[pix_id * K + layer_cnt] = collected_id[j];
                layer_cnt++;
                if (layer_cnt >= K) {
                    done = true;
                    continue;
                }
            }
            else {
                if (layer_cnt < K) {
                    final_idx[pix_id * K + layer_cnt] = collected_id[j];
                    layer_cnt++;
                }
            }
        }
    }

    if (inside) {
        final_T[pix_id] = T;
        ncontrib[pix_id] = last_contributor;
        for (int k = 0; k < c_num; k++)
            // // bg only for RGB: the first 3 channels
            // if (k < 3)
            //     rendered_feature[k * H * W + pix_id] = F[k] + T * bg;
            // else
            //     rendered_feature[k * H * W + pix_id] = F[k] + T * 0.0;
            rendered_feature[k * H * W + pix_id] = F[k] + T * bg;
    }
}

template <uint32_t CNum>
__global__ void __launch_bounds__(BLOCK_X *BLOCK_Y)
    renderGaussianEllipseBackwardCUDAKernel(const int P,
                                    const float2 *__restrict__ uv,
                                    const float3 *__restrict__ conic,
                                    const float *__restrict__ opacity,
                                    const float *__restrict__ feature,
                                    const int *__restrict__ idx_sorted,
                                    const int2 *__restrict__ tile_range,
                                    const float bg,
                                    const int C,
                                    const int W,
                                    const int H,
                                    float *__restrict__ final_T,
                                    int *__restrict__ ncontrib,
                                    const float *__restrict__ dL_drendered,
                                    float2 *__restrict__ dL_duv,
                                    float2 *__restrict__ dL_dabs_uv,
                                    float3 *__restrict__ dL_dconic,
                                    float *__restrict__ dL_dopacity,
                                    float *__restrict__ dL_dfeature) {
    auto block = cg::this_thread_block();
    int32_t tile_grid_x = (W + BLOCK_X - 1) / BLOCK_X;
    int32_t tile_id =
        block.group_index().y * tile_grid_x + block.group_index().x;
    uint2 pix = {block.group_index().x * BLOCK_X + block.thread_index().x,
                 block.group_index().y * BLOCK_Y + block.thread_index().y};
    uint32_t pix_id = W * pix.y + pix.x;
    float2 pixf = {(float)pix.x, (float)pix.y};
    const int c_num = min(CNum, C);

    bool inside = pix.x < W && pix.y < H;
    bool done = !inside;

    int2 range = tile_range[tile_id];
    const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
    int toDo = range.y - range.x;

    __shared__ int collected_id[BLOCK_SIZE];
    __shared__ float2 collected_uv[BLOCK_SIZE];
    __shared__ float3 collected_conic[BLOCK_SIZE];
    __shared__ float collected_opacity[BLOCK_SIZE];
    __shared__ float collected_feature[CNum * BLOCK_SIZE];

    const float T_final = inside ? final_T[pix_id] : 0;
    float T = T_final;

    uint32_t contributor = toDo;
    const int last_contributor = inside ? ncontrib[pix_id] : 0;

    float accum_rec[CNum] = {0};
    float dL_dpixel[CNum] = {0};
    if (inside)
        for (int ch = 0; ch < c_num; ch++)
            dL_dpixel[ch] = dL_drendered[ch * H * W + pix_id];

    float last_alpha = 0;
    float last_feature[CNum] = {0};
    for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE) {
        block.sync();
        const int progress = i * BLOCK_SIZE + block.thread_rank();
        if (range.x + progress < range.y) {
            const int coll_id = idx_sorted[range.y - progress - 1];
            collected_id[block.thread_rank()] = coll_id;
            collected_uv[block.thread_rank()] = uv[coll_id];
            collected_conic[block.thread_rank()] = conic[coll_id];
            collected_opacity[block.thread_rank()] = opacity[coll_id];
            for (int ch = 0; ch < c_num; ch++)
                collected_feature[ch * BLOCK_SIZE + block.thread_rank()] =
                    feature[ch * P + coll_id];
        }
        block.sync();

        for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++) {
            contributor--;
            if (contributor >= last_contributor)
                continue;

            float2 vec = {collected_uv[j].x - pixf.x,
                          collected_uv[j].y - pixf.y};

            // Gabor 核计算
            float2 dx = vec;
            float theta = 0.5f; // 方向
            float cosr = cos(theta);
            float sinr = sin(theta);
            
            // 转换坐标到旋转空间
            float x_theta = dx.x * cosr + dx.y * sinr;
            float y_theta = -dx.x * sinr + dx.y * cosr;
            
            // 高斯部分计算
            float gaussian_part = -0.5f * (collected_conic[j].x * vec.x * vec.x +
                                   collected_conic[j].z * vec.y * vec.y) -
                          collected_conic[j].y * vec.x * vec.y;
            
            // 跳过高斯部分较弱的点
            if (gaussian_part > 0.0f)
                continue;
            
            float g_factor = exp(gaussian_part);
            
            // 使用与提供代码相同的Gabor条纹计算
            float sinusoid_part = 0.0f;
            for(int w_idx = 0; w_idx < SELECTED_NUM_FREQUENCIES; w_idx++){
                float f = max_frequency * (w_idx + 1) / (float)TOTAL_NUM_FREQUENCIES;
                sinusoid_part += 0.5f + 0.5f * cos(f * x_theta);
            }
            sinusoid_part /= SELECTED_NUM_FREQUENCIES; // 归一化
            
            // 直接使用计算的sinusoid_part作为调制因子
            const float alpha = min(0.99f, collected_opacity[j] * g_factor * sinusoid_part);
            if (alpha < 1.0 / 255.0)
                continue;

            T = T / (1.f - alpha);
            const float dchannel_dcolor = alpha * T;

            float dL_dalpha = 0.0f;
            const int global_id = collected_id[j];
            for (int ch = 0; ch < c_num; ch++) {
                const float current_feature =
                    collected_feature[ch * BLOCK_SIZE + j];
                accum_rec[ch] = last_alpha * last_feature[ch] +
                                (1.f - last_alpha) * accum_rec[ch];
                last_feature[ch] = current_feature;

                dL_dalpha += (current_feature - accum_rec[ch]) * dL_dpixel[ch];
                atomicAdd(&dL_dfeature[ch * P + global_id],
                          dchannel_dcolor * dL_dpixel[ch]);
            }

            dL_dalpha *= T;
            last_alpha = alpha;

            float bg_dot_dpixel = 0;
            for (int ch = 0; ch < c_num; ch++)
                bg_dot_dpixel += bg * dL_dpixel[ch];

            dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;

            // 计算对g_factor和sinusoid_part的导数
            const float dL_dg_factor = collected_opacity[j] * sinusoid_part * dL_dalpha;
            const float dL_dsinusoid = collected_opacity[j] * g_factor * dL_dalpha;
            
            // 计算对每个频率分量的导数
            float dL_dx_theta = 0.0f;
            for(int w_idx = 0; w_idx < SELECTED_NUM_FREQUENCIES; w_idx++){
                float f = max_frequency * (w_idx + 1) / (float)TOTAL_NUM_FREQUENCIES;
                // 对cos(f*x_theta)求导得到-f*sin(f*x_theta)
                dL_dx_theta += dL_dsinusoid * 0.5f * (-f) * sin(f * x_theta) / SELECTED_NUM_FREQUENCIES;
            }
            
            // 计算旋转坐标对原始坐标的导数
            float dx_theta_dvecx = cos(theta);
            float dx_theta_dvecy = sin(theta);
            
            // 计算高斯部分对向量的导数
            float dpower_dvecx = -collected_conic[j].x * vec.x - collected_conic[j].y * vec.y;
            float dpower_dvecy = -collected_conic[j].z * vec.y - collected_conic[j].y * vec.x;
            
            // 链式法则：组合导数
            float dg_factor_dvecx = g_factor * dpower_dvecx;
            float dg_factor_dvecy = g_factor * dpower_dvecy;
            
            float dL_dvecx = dL_dg_factor * dg_factor_dvecx + dL_dx_theta * dx_theta_dvecx;
            float dL_dvecy = dL_dg_factor * dg_factor_dvecy + dL_dx_theta * dx_theta_dvecy;
            
            const float2 dL_dvec = {dL_dvecx, dL_dvecy};

            atomicAdd(&dL_duv[global_id].x, dL_dvec.x);
            atomicAdd(&dL_duv[global_id].y, dL_dvec.y);
            atomicAdd(&dL_dabs_uv[global_id].x, fabsf(dL_dvec.x));
            atomicAdd(&dL_dabs_uv[global_id].y, fabsf(dL_dvec.y));
            
            // 更新对 conic 参数的导数
            float dconic_x = -0.5f * vec.x * vec.x;
            float dconic_y = -vec.x * vec.y;
            float dconic_z = -0.5f * vec.y * vec.y;
            
            atomicAdd(&dL_dconic[global_id].x, dL_dg_factor * g_factor * dconic_x);
            atomicAdd(&dL_dconic[global_id].y, dL_dg_factor * g_factor * dconic_y);
            atomicAdd(&dL_dconic[global_id].z, dL_dg_factor * g_factor * dconic_z);
            
            // 对透明度的导数
            atomicAdd(&dL_dopacity[global_id], g_factor * sinusoid_part * dL_dalpha);
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
renderGaussianEllipseForward(const torch::Tensor &uv,
                     const torch::Tensor &conic,
                     const torch::Tensor &opacity,
                     const torch::Tensor &feature,
                     const torch::Tensor &idx_sorted,
                     const torch::Tensor &tile_range,
                     const float bg,
                     const int W,
                     const int H,
                     const int K,
                     const bool enable_truncation) {
    CHECK_INPUT(uv);
    CHECK_INPUT(conic);
    CHECK_INPUT(opacity);
    CHECK_INPUT(feature);
    CHECK_INPUT(idx_sorted);
    CHECK_INPUT(tile_range);

    const int P = feature.size(0);
    const int C = feature.size(1);

    auto int_opts = feature.options().dtype(torch::kInt32);
    auto float_opts = feature.options().dtype(torch::kFloat32);
    torch::Tensor rendered_feature = torch::zeros({C, H, W}, float_opts);
    torch::Tensor final_T = torch::zeros({H, W}, float_opts);
    torch::Tensor ncontrib = torch::zeros({H, W}, int_opts);
    torch::Tensor final_idx = torch::zeros({H, W, K}, int_opts) - 1;

    const dim3 tile_grid(
        (W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, 1);
    const dim3 block(BLOCK_X, BLOCK_Y, 1);

    // Transpose to a [C, N] tensor for scalable feature implementation.
    torch::Tensor feature_permute = feature.transpose(0, 1);

    // Select the optimal template kernel based on channel number.
    // If the number exceed the template's limit, process channels in sequential
    // batches. ToDo: Find a better way to do this.
    for (int C0 = 0; C0 < C;) {
        size_t p_data_offset = C0 * P;
        size_t img_data_offset = C0 * H * W;

        if (C - C0 <= 3) {
            renderGaussianEllipseForwardCUDAKernel<3><<<tile_grid, block>>>(
                P,
                (float2 *)uv.contiguous().data_ptr<float>(),
                (float3 *)conic.contiguous().data_ptr<float>(),
                opacity.contiguous().data_ptr<float>(),
                feature_permute.contiguous().data_ptr<float>() + p_data_offset,
                idx_sorted.contiguous().data_ptr<int>(),
                (int2 *)tile_range.contiguous().data_ptr<int>(),
                bg,
                C - C0,
                W,
                H,
                K,
                enable_truncation,
                final_T.data_ptr<float>(),
                ncontrib.data_ptr<int>(),
                final_idx.data_ptr<int>(),
                rendered_feature.data_ptr<float>() + img_data_offset);
            C0 += 3;
        } else if (C - C0 <= 6) {
            renderGaussianEllipseForwardCUDAKernel<6><<<tile_grid, block>>>(
                P,
                (float2 *)uv.contiguous().data_ptr<float>(),
                (float3 *)conic.contiguous().data_ptr<float>(),
                opacity.contiguous().data_ptr<float>(),
                feature_permute.contiguous().data_ptr<float>() + p_data_offset,
                idx_sorted.contiguous().data_ptr<int>(),
                (int2 *)tile_range.contiguous().data_ptr<int>(),
                bg,
                C - C0,
                W,
                H,
                K,
                enable_truncation,
                final_T.data_ptr<float>(),
                ncontrib.data_ptr<int>(),
                final_idx.data_ptr<int>(),
                rendered_feature.data_ptr<float>() + img_data_offset);
            C0 += 6;
        } else if (C - C0 <= 12) {
            renderGaussianEllipseForwardCUDAKernel<12><<<tile_grid, block>>>(
                P,
                (float2 *)uv.contiguous().data_ptr<float>(),
                (float3 *)conic.contiguous().data_ptr<float>(),
                opacity.contiguous().data_ptr<float>(),
                feature_permute.contiguous().data_ptr<float>() + p_data_offset,
                idx_sorted.contiguous().data_ptr<int>(),
                (int2 *)tile_range.contiguous().data_ptr<int>(),
                bg,
                C - C0,
                W,
                H,
                K,
                enable_truncation,
                final_T.data_ptr<float>(),
                ncontrib.data_ptr<int>(),
                final_idx.data_ptr<int>(),
                rendered_feature.data_ptr<float>() + img_data_offset);
            C0 += 12;
        } else if (C - C0 <= 18) {
            renderGaussianEllipseForwardCUDAKernel<18><<<tile_grid, block>>>(
                P,
                (float2 *)uv.contiguous().data_ptr<float>(),
                (float3 *)conic.contiguous().data_ptr<float>(),
                opacity.contiguous().data_ptr<float>(),
                feature_permute.contiguous().data_ptr<float>() + p_data_offset,
                idx_sorted.contiguous().data_ptr<int>(),
                (int2 *)tile_range.contiguous().data_ptr<int>(),
                bg,
                C - C0,
                W,
                H,
                K,
                enable_truncation,
                final_T.data_ptr<float>(),
                ncontrib.data_ptr<int>(),
                final_idx.data_ptr<int>(),
                rendered_feature.data_ptr<float>() + img_data_offset);
            C0 += 18;
        } else if (C - C0 <= 24) {
            renderGaussianEllipseForwardCUDAKernel<24><<<tile_grid, block>>>(
                P,
                (float2 *)uv.contiguous().data_ptr<float>(),
                (float3 *)conic.contiguous().data_ptr<float>(),
                opacity.contiguous().data_ptr<float>(),
                feature_permute.contiguous().data_ptr<float>() + p_data_offset,
                idx_sorted.contiguous().data_ptr<int>(),
                (int2 *)tile_range.contiguous().data_ptr<int>(),
                bg,
                C - C0,
                W,
                H,
                K,
                enable_truncation,
                final_T.data_ptr<float>(),
                ncontrib.data_ptr<int>(),
                final_idx.data_ptr<int>(),
                rendered_feature.data_ptr<float>() + img_data_offset);
            C0 += 24;
        } else {
            renderGaussianEllipseForwardCUDAKernel<32><<<tile_grid, block>>>(
                P,
                (float2 *)uv.contiguous().data_ptr<float>(),
                (float3 *)conic.contiguous().data_ptr<float>(),
                opacity.contiguous().data_ptr<float>(),
                feature_permute.contiguous().data_ptr<float>() + p_data_offset,
                idx_sorted.contiguous().data_ptr<int>(),
                (int2 *)tile_range.contiguous().data_ptr<int>(),
                bg,
                C - C0,
                W,
                H,
                K,
                enable_truncation,
                final_T.data_ptr<float>(),
                ncontrib.data_ptr<int>(),
                final_idx.data_ptr<int>(),
                rendered_feature.data_ptr<float>() + img_data_offset);
            C0 += 32;
        }
    }

    return std::make_tuple(rendered_feature, final_T, ncontrib, final_idx);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
renderGaussianEllipseBackward(const torch::Tensor &uv,
                      const torch::Tensor &conic,
                      const torch::Tensor &opacity,
                      const torch::Tensor &feature,
                      const torch::Tensor &idx_sorted,
                      const torch::Tensor &tile_range,
                      const float bg,
                      const int W,
                      const int H,
                      const torch::Tensor &final_T,
                      const torch::Tensor &ncontrib,
                      const torch::Tensor &dL_drendered) {
    CHECK_INPUT(uv);
    CHECK_INPUT(conic);
    CHECK_INPUT(opacity);
    CHECK_INPUT(feature);
    CHECK_INPUT(idx_sorted);
    CHECK_INPUT(tile_range);
    CHECK_INPUT(final_T);
    CHECK_INPUT(ncontrib);
    CHECK_INPUT(dL_drendered);

    const int P = feature.size(0);
    const int C = feature.size(1);

    auto float_opts = feature.options().dtype(torch::kFloat32);
    torch::Tensor dL_duv = torch::zeros({P, 2}, float_opts);
    torch::Tensor dL_dabs_uv = torch::zeros({P, 2}, float_opts);
    torch::Tensor dL_dconic = torch::zeros({P, 3}, float_opts);
    torch::Tensor dL_dopacity = torch::zeros({P, 1}, float_opts);
    torch::Tensor dL_dfeature_permute = torch::zeros({C, P}, float_opts);
    torch::Tensor dL_dalpha = torch::zeros({H, W}, float_opts);

    const dim3 tile_grid(
        (W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, 1);
    const dim3 block(BLOCK_X, BLOCK_Y, 1);

    // [C, N]
    torch::Tensor feature_permute = feature.transpose(0, 1);

    for (int C0 = 0; C0 < C;) {
        size_t p_data_offset = C0 * P;
        size_t img_data_offset = C0 * H * W;

        if (C - C0 <= 3) {
            renderGaussianEllipseBackwardCUDAKernel<3><<<tile_grid, block>>>(
                P,
                (float2 *)uv.contiguous().data_ptr<float>(),
                (float3 *)conic.contiguous().data_ptr<float>(),
                opacity.contiguous().data_ptr<float>(),
                feature_permute.contiguous().data_ptr<float>() + p_data_offset,
                idx_sorted.contiguous().data_ptr<int>(),
                (int2 *)tile_range.contiguous().data_ptr<int>(),
                bg,
                C - C0,
                W,
                H,
                final_T.contiguous().data_ptr<float>(),
                ncontrib.contiguous().data_ptr<int>(),
                dL_drendered.contiguous().data_ptr<float>() + img_data_offset,
                (float2 *)dL_duv.data_ptr<float>(),
                (float2 *)dL_dabs_uv.data_ptr<float>(),
                (float3 *)dL_dconic.data_ptr<float>(),
                dL_dopacity.data_ptr<float>(),
                dL_dfeature_permute.data_ptr<float>() + p_data_offset);
            C0 += 3;
        } else if (C - C0 <= 6) {
            renderGaussianEllipseBackwardCUDAKernel<6><<<tile_grid, block>>>(
                P,
                (float2 *)uv.contiguous().data_ptr<float>(),
                (float3 *)conic.contiguous().data_ptr<float>(),
                opacity.contiguous().data_ptr<float>(),
                feature_permute.contiguous().data_ptr<float>() + p_data_offset,
                idx_sorted.contiguous().data_ptr<int>(),
                (int2 *)tile_range.contiguous().data_ptr<int>(),
                bg,
                C - C0,
                W,
                H,
                final_T.contiguous().data_ptr<float>(),
                ncontrib.contiguous().data_ptr<int>(),
                dL_drendered.contiguous().data_ptr<float>() + img_data_offset,
                (float2 *)dL_duv.data_ptr<float>(),
                (float2 *)dL_dabs_uv.data_ptr<float>(),
                (float3 *)dL_dconic.data_ptr<float>(),
                dL_dopacity.data_ptr<float>(),
                dL_dfeature_permute.data_ptr<float>() + p_data_offset);
            C0 += 6;
        } else if (C - C0 <= 12) {
            renderGaussianEllipseBackwardCUDAKernel<12><<<tile_grid, block>>>(
                P,
                (float2 *)uv.contiguous().data_ptr<float>(),
                (float3 *)conic.contiguous().data_ptr<float>(),
                opacity.contiguous().data_ptr<float>(),
                feature_permute.contiguous().data_ptr<float>() + p_data_offset,
                idx_sorted.contiguous().data_ptr<int>(),
                (int2 *)tile_range.contiguous().data_ptr<int>(),
                bg,
                C - C0,
                W,
                H,
                final_T.contiguous().data_ptr<float>(),
                ncontrib.contiguous().data_ptr<int>(),
                dL_drendered.contiguous().data_ptr<float>() + img_data_offset,
                (float2 *)dL_duv.data_ptr<float>(),
                (float2 *)dL_dabs_uv.data_ptr<float>(),
                (float3 *)dL_dconic.data_ptr<float>(),
                dL_dopacity.data_ptr<float>(),
                dL_dfeature_permute.data_ptr<float>() + p_data_offset);
            C0 += 12;
        } else if (C - C0 <= 18) {
            renderGaussianEllipseBackwardCUDAKernel<18><<<tile_grid, block>>>(
                P,
                (float2 *)uv.contiguous().data_ptr<float>(),
                (float3 *)conic.contiguous().data_ptr<float>(),
                opacity.contiguous().data_ptr<float>(),
                feature_permute.contiguous().data_ptr<float>() + p_data_offset,
                idx_sorted.contiguous().data_ptr<int>(),
                (int2 *)tile_range.contiguous().data_ptr<int>(),
                bg,
                C - C0,
                W,
                H,
                final_T.contiguous().data_ptr<float>(),
                ncontrib.contiguous().data_ptr<int>(),
                dL_drendered.contiguous().data_ptr<float>() + img_data_offset,
                (float2 *)dL_duv.data_ptr<float>(),
                (float2 *)dL_dabs_uv.data_ptr<float>(),
                (float3 *)dL_dconic.data_ptr<float>(),
                dL_dopacity.data_ptr<float>(),
                dL_dfeature_permute.data_ptr<float>() + p_data_offset);
            C0 += 18;
        } else if (C - C0 <= 24) {
            renderGaussianEllipseBackwardCUDAKernel<24><<<tile_grid, block>>>(
                P,
                (float2 *)uv.contiguous().data_ptr<float>(),
                (float3 *)conic.contiguous().data_ptr<float>(),
                opacity.contiguous().data_ptr<float>(),
                feature_permute.contiguous().data_ptr<float>() + p_data_offset,
                idx_sorted.contiguous().data_ptr<int>(),
                (int2 *)tile_range.contiguous().data_ptr<int>(),
                bg,
                C - C0,
                W,
                H,
                final_T.contiguous().data_ptr<float>(),
                ncontrib.contiguous().data_ptr<int>(),
                dL_drendered.contiguous().data_ptr<float>() + img_data_offset,
                (float2 *)dL_duv.data_ptr<float>(),
                (float2 *)dL_dabs_uv.data_ptr<float>(),
                (float3 *)dL_dconic.data_ptr<float>(),
                dL_dopacity.data_ptr<float>(),
                dL_dfeature_permute.data_ptr<float>() + p_data_offset);
            C0 += 24;
        } else {
            renderGaussianEllipseBackwardCUDAKernel<32><<<tile_grid, block>>>(
                P,
                (float2 *)uv.contiguous().data_ptr<float>(),
                (float3 *)conic.contiguous().data_ptr<float>(),
                opacity.contiguous().data_ptr<float>(),
                feature_permute.contiguous().data_ptr<float>() + p_data_offset,
                idx_sorted.contiguous().data_ptr<int>(),
                (int2 *)tile_range.contiguous().data_ptr<int>(),
                bg,
                C - C0,
                W,
                H,
                final_T.contiguous().data_ptr<float>(),
                ncontrib.contiguous().data_ptr<int>(),
                dL_drendered.contiguous().data_ptr<float>() + img_data_offset,
                (float2 *)dL_duv.data_ptr<float>(),
                (float2 *)dL_dabs_uv.data_ptr<float>(),
                (float3 *)dL_dconic.data_ptr<float>(),
                dL_dopacity.data_ptr<float>(),
                dL_dfeature_permute.data_ptr<float>() + p_data_offset);
            C0 += 32;
        }
    }

    // [N, C]
    torch::Tensor dL_dfeature = dL_dfeature_permute.transpose(0, 1);

    return std::make_tuple(dL_duv, dL_dconic, dL_dopacity, dL_dfeature, dL_dabs_uv);
}