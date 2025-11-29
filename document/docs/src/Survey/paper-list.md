# 3DGS-SLAM Paper List

<style>
body.special-layout .md-grid {
  max-width: 1960px !important;
}
.md-sidebar--primary {
  width: 11rem;
  left: -11rem;
}
</style>

<script>
  document.body.classList.add('special-layout');
</script>

<table>
  <thead>
    <tr>
      <th scope="col" style="text-align: center;">Venue</th>
      <th scope="col" style="text-align: center;">Paper Abbr</th>
      <th scope="col" style="text-align: center;">Title</th>
      <th scope="col" style="text-align: center;">Equipment</th>
      <th scope="col" style="text-align: center;">External Tracker</th>
      <th scope="col" style="text-align: center;">Extra Priors</th>
      <th scope="col" style="text-align: center;">Code</th>
    </tr>
  </thead>
  <tbody>

    <tr>
      <th scope="row" style="text-align: center;">CVPR'24</th>
      <td style="text-align: center;"> SplaTAM </td>
      <td>
        <a href="https://openaccess.thecvf.com/content/CVPR2024/papers/Keetha_SplaTAM_Splat_Track__Map_3D_Gaussians_for_Dense_RGB-D_CVPR_2024_paper.pdf" target="_blank" style="text-decoration: none; color:rgb(28, 30, 39); font-weight: 500;"> 
          SplaTAM: Splat, Track & Map 3D Gaussians for Dense RGB-D SLAM
        </a>
      </td>
      <td style="text-align: center;"> RGB-D </td>
      <td style="text-align: center;"></td>
      <td style="text-align: center;"></td>
      <td style="text-align: center;">
        <a href="https://github.com/spla-tam/SplaTAM" target="_blank" style="text-decoration: none;">
            <img src="https://img.shields.io/github/stars/spla-tam/SplaTAM?style=for-the-badge&logo=github&logoColor=5865F2&color=E0F2FE&labelColor=FAFAFA" 
                alt="GitHub stars" 
                style="height: 28px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
        </a>
      </td>
    </tr>

    <tr>
      <th scope="row" style="text-align: center;">CVPR'24</th>
      <td style="text-align: center;"> Photo-SLAM </td>
      <td>
        <a href="https://openaccess.thecvf.com/content/CVPR2024/papers/Huang_Photo-SLAM_Real-time_Simultaneous_Localization_and_Photorealistic_Mapping_for_Monocular_Stereo_CVPR_2024_paper.pdf" target="_blank" style="text-decoration: none; color:rgb(28, 30, 39); font-weight: 500;"> 
          Photo-SLAM: Real-time Simultaneous Localization and Photorealistic Mapping for Monocular, Stereo, and RGB-D Cameras
        </a>
      </td>
      <td style="text-align: center;"> RGB-D, RGB, Stereo </td>
      <td style="text-align: center;"> ORB-SLAM3 </td>
      <td style="text-align: center;"></td>
      <td style="text-align: center;">
        <a href="https://github.com/HuajianUP/Photo-SLAM" target="_blank" style="text-decoration: none;">
            <img src="https://img.shields.io/github/stars/HuajianUP/Photo-SLAM?style=for-the-badge&logo=github&logoColor=5865F2&color=E0F2FE&labelColor=FAFAFA" 
                alt="GitHub stars" 
                style="height: 28px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
        </a>
      </td>
      <!-- <td>
        <a href="https://huajianup.github.io/research/Photo-SLAM" 
          target="_blank" 
          style="text-decoration: none;">
          <img src="https://img.shields.io/badge/Website-5865F2?style=for-the-badge&logoColor=5865F2&color=E0F2FE&labelColor=FFFFFF" 
              alt="Website" 
              style="height: 28px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
        </a>
      </td> -->
    </tr>

    <tr>
      <th scope="row" style="text-align: center;">ECCV'24</th>
      <td style="text-align: center;"> GS-ICP SLAM </td>
      <td>
        <a href="https://arxiv.org/pdf/2403.12550" target="_blank" style="text-decoration: none; color:rgb(28, 30, 39); font-weight: 500;"> 
          RGBD GS-ICP SLAM
        </a>
      </td>
      <td style="text-align: center;"> RGB-D </td>
      <td style="text-align: center;"> G-ICP </td>
      <td style="text-align: center;"></td>
      <td style="text-align: center;">
        <a href="https://github.com/Lab-of-AI-and-Robotics/GS_ICP_SLAM" target="_blank" style="text-decoration: none;">
            <img src="https://img.shields.io/github/stars/Lab-of-AI-and-Robotics/GS_ICP_SLAM?style=for-the-badge&logo=github&logoColor=5865F2&color=E0F2FE&labelColor=FAFAFA" 
                alt="GitHub stars" 
                style="height: 28px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
        </a>
      </td>
    </tr>

    <tr>
      <th scope="row" style="text-align: center;"> CVPR'25 </th>
      <td style="text-align: center;"> WildGS-SLAM </td>
      <td>
        <a href="https://arxiv.org/pdf/2504.03886" target="_blank" style="text-decoration: none; color:rgb(28, 30, 39); font-weight: 500;"> 
          WildGS-SLAM: Monocular Gaussian Splatting SLAM in Dynamic Environments
        </a>
      </td>
      <td style="text-align: center;"> Monocular RGB </td>
      <td style="text-align: center;"> Depth & Uncertainty Guided DBA </td>
      <td style="text-align: center;"> Uncertainty Prediction </td>
      <td style="text-align: center;">
        <a href="https://github.com/GradientSpaces/WildGS-SLAM" target="_blank" style="text-decoration: none;">
            <img src="https://img.shields.io/github/stars/GradientSpaces/WildGS-SLAM?style=for-the-badge&logo=github&logoColor=5865F2&color=E0F2FE&labelColor=FAFAFA" 
                alt="GitHub stars" 
                style="height: 28px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
        </a>
      </td>
    </tr>

    <tr>
      <th scope="row" style="text-align: center;">ICCV'25</th>
      <td style="text-align: center;"> SEGS-SLAM </td>
      <td>
        <a href="https://arxiv.org/pdf/2501.05242" target="_blank" style="text-decoration: none; color:rgb(28, 30, 39); font-weight: 500;"> 
          SEGS-SLAM: Structure-enhanced 3D Gaussian Splatting SLAM with Appearance Embedding
        </a>
      </td>
      <td style="text-align: center;"> Monocular, Stereo, RGB-D </td>
      <td style="text-align: center;"> ORB-SLAM3 </td>
      <td style="text-align: center;">  </td>
      <td style="text-align: center;">
        <a href="https://github.com/leaner-forever/SEGS-SLAM" target="_blank" style="text-decoration: none;">
            <img src="https://img.shields.io/github/stars/leaner-forever/SEGS-SLAM?style=for-the-badge&logo=github&logoColor=5865F2&color=E0F2FE&labelColor=FAFAFA" 
                alt="GitHub stars" 
                style="height: 28px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
        </a>
      </td>
    </tr>

    <tr>
      <th scope="row" style="text-align: center;">ICRA'25</th>
      <td style="text-align: center;"> OpenGS-SLAM </td>
      <td>
        <a href="https://arxiv.org/pdf/2502.15633" target="_blank" style="text-decoration: none; color:rgb(28, 30, 39); font-weight: 500;">
          RGB-Only Gaussian Splatting SLAM for Unbounded Outdoor Scenes
        </a>
      </td>
      <td style="text-align: center;"> RGB </td>
      <td style="text-align: center;"> </td>
      <td style="text-align: center;"> pointmap regression network </td>
      <td style="text-align: center;">
        <a href="https://github.com/3DAgentWorld/OpenGS-SLAM" target="_blank" style="text-decoration: none;">
            <img src="https://img.shields.io/github/stars/3DAgentWorld/OpenGS-SLAM?style=for-the-badge&logo=github&logoColor=5865F2&color=E0F2FE&labelColor=FAFAFA" 
                alt="GitHub stars" 
                style="height: 28px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
        </a>
      </td>
    </tr>

    <tr>
      <th scope="row" style="text-align: center;">ICCV'25</th>
      <td style="text-align: center;"> S3PO-GS SLAM </td>
      <td>
        <a href="https://arxiv.org/pdf/2507.03737" target="_blank" style="text-decoration: none; color:rgb(28, 30, 39); font-weight: 500;">
          Outdoor Monocular SLAM with Global Scale-Consistent 3D Gaussian Pointmaps
        </a>
      </td>
      <td style="text-align: center;"> RGB </td>
      <td style="text-align: center;"> </td>
      <td style="text-align: center;"> pre-trained pointmap model </td>
      <td style="text-align: center;">
        <a href="https://github.com/3DAgentWorld/S3PO-GS" target="_blank" style="text-decoration: none;">
            <img src="https://img.shields.io/github/stars/3DAgentWorld/S3PO-GS?style=for-the-badge&logo=github&logoColor=5865F2&color=E0F2FE&labelColor=FAFAFA" 
                alt="GitHub stars" 
                style="height: 28px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
        </a>
      </td>
    </tr>

    <tr>
      <th scope="row" style="text-align: center;">arXiv'25</th>
      <td style="text-align: center;"> 4D Gaussian Splatting SLAM </td>
      <td>
        <a href="https://arxiv.org/pdf/2503.16710" target="_blank" style="text-decoration: none; color:rgb(28, 30, 39); font-weight: 500;"> 
          4D Gaussian Splatting SLAM
        </a>
      </td>
      <td style="text-align: center;">  </td>
      <td style="text-align: center;">  </td>
      <td style="text-align: center;"></td>
      <td style="text-align: center;">
        <a href="https://github.com/yanyan-li/4DGS-SLAM" target="_blank" style="text-decoration: none;">
            <img src="https://img.shields.io/github/stars/yanyan-li/4DGS-SLAM?style=for-the-badge&logo=github&logoColor=5865F2&color=E0F2FE&labelColor=FAFAFA" 
                alt="GitHub stars" 
                style="height: 28px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
        </a>
      </td>
    </tr>

  </tbody>
</table>