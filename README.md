
<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]




<!-- PROJECT LOGO -->
<br />

<div align="center">
  <a href="https://github.com/3C-SCSU/Avatar">
    <img src="https://avatars.githubusercontent.com/u/114175379?v=4" alt="Logo" width="80" height="80">
  </a>
  <h1 align="center">Avatar</h1>



  <p align="center">
    An application of Machine Learning on brainwave donations in order to remotely pilot a drone in real-time through brainwave reading. 
    <br />
    <br />
    <a href="https://github.com/3C-SCSU/Avatar"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/3C-SCSU/Avatar/wiki"><strong>Get help in the Wiki »</strong></a>
    <br />
    <br />
<a href="https://github.com/3C-SCSU/Avatar">View Demo</a>
    ·
    <a href="https://github.com/3C-SCSU/Avatar/issues">Report Bug</a>
    ·
    <a href="https://github.com/3C-SCSU/Avatar/issues">Request Feature</a>    

  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

<!--[![Product Name Screen Shot][product-screenshot]](https://example.com)-->

This is an open source project from the Cloud Computing Club of Saint Cloud State University. Brainwaves are read using an OpenBCI headset. These brainwaves are sent to a server and randomly renamed and dated for privacy. After shuffling, the data is loaded into Spark where it is processed and a piloting prediction is made. A client-side request reads the most recent prediction and instructs the drone for its next action. 

 ### Institutional Review Board

Brainwave collection is considered human subject donation, and this project is compliant with the policies of the <a href="https://www.stcloudstate.edu/irb/">IRB </a>. 



<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With
<!--   ********TODO: UPDATE  BUIlT WITH SECTION ****************
* [![Next][Next.js]][Next-url]
* [![React][React.js]][React-url]
* [![Vue][Vue.js]][Vue-url]
* [![Angular][Angular.io]][Angular-url]
* [![Svelte][Svelte.dev]][Svelte-url]
* [![Laravel][Laravel.com]][Laravel-url]
* [![Bootstrap][Bootstrap.com]][Bootstrap-url]
* [![JQuery][JQuery.com]][JQuery-url]
***************END OF UPDATE INSTALLTION INFORMATION    --->
<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

This section will be updated further as the project progresses. 

### Prerequisites
<h4> Equipment</h4>

<a href="https://shop.openbci.com/products/ultracortex-mark-iv" > OpenBCI Headset </a> with <a href="https://shop.openbci.com/products/cyton-daisy-biosensing-boards-16-channel" > Cyton 16 Channel board </a>

<a href = "https://www.amazon.com/DJI-CP-TL-00000026-02-Tello-EDU/dp/B07TZG2NNT?ref_=ast_sto_dp" > DJI Tello Edu Drone </a> 


<!---   ***************TODO: UPDATE INSTALLATION INFORMATION 
This is an example of how to list things you need to use the software and how to install them.
* npm
  ```sh
  npm install npm@latest -g
  ```

### Installation

1. Get a free API Key at [https://example.com](https://example.com)
2. Clone the repo
   ```sh
   git clone https://github.com/3C-SCSU/Avatar.git
   ```
3. Install NPM packages
   ```sh
   npm install
   ```
4. Enter your API in `config.js`
   ```js
   const API_KEY = 'ENTER YOUR API';
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

***************END OF UPDATE INSTALLATION INFORMATION    --->

<!-- USAGE EXAMPLES -->
## Usage

This section will be updated further as the project progresses.

<!---   ***************TODO: UPDATE USAGE INFORMATION 
_For more examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#readme-top">back to top</a>)</p>

***************END OF UPDATE USAGE INFORMATION    --->

<!-- ROADMAP -->
## Roadmap

- [x] Read Brainwaves using OpenBCI software
	- [ ] Configure workstation Chromebooks for compatibility with BCI dongle
- [x] Encrypted File Transfer from workstation to VPS 
- [ ] Automated file shuffling
- [x] Containerize VPS applications 
- [ ] Implement K8s
- [x] Connect Spark with GCP bucket
- [ ] Configure Spark with Delta Lake
- [ ] Create Application to Control Drone
	- [ ] GUI
	- [x] Drone Control
	- [ ] Headset Data transferring
	- [ ] Connect with ML Predictions

See the [open issues](https://github.com/3C-SCSU/Avatar/issues) for a full list of proposed features (and known issues). 

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

### University Contributions
If you have been informed that your submission for contribution was selected and approved by the Cloud Computing Club, follow the below steps to add it to the project and receive the bounty. 
1. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
2. Name the feature branch using format `feature-bounty` ex. `file_shuffler-bounty` 
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request and select a review from the list of Collaborators.
6. Update the  <a href="https://github.com/3C-SCSU/Avatar/wiki">Wiki </a> with an explanation of your added feature. 


### External Contributions
Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


Pull requests will be rieviewed and merged: 

### Community Members by Category

Members by promotion and merit voted in the weekly's #3C meeting:

1. Sporadic contributors: any developer who decides to start contributing to the project's code base.

2. Committers: contributors promoted above the so-called sporadic contributors. To become a committer, the person must get 20 patches approved. When becoming a committer, then the committer will review and support code submitted from sporadic contributors

3. Reviewers: are generally either the founding developers of the project - like initial administrator developers; or contributors who become committers, who then, after 50 patch contributions to the project's code base, are promoted to the status of reviewers. Reviewers will decide and review any conflict in code contributions to the project.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Cloud Computing Club 3c.scsu@gmail.com

Discord: https://discord.gg/mxbQ7HpPjq

Project Link: [https://github.com/3C-SCSU/Avatar](https://github.com/3C-SCSU/Avatar)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* []() Dr. A. Cavalcanti - Faculty Advisor for #3C-SCSU


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/3C-SCSU/Avatar.svg?style=for-the-badge
[contributors-url]: https://github.com/3C-SCSU/Avatar/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/3C-SCSU/Avatar.svg?style=for-the-badge
[forks-url]: https://github.com/3C-SCSU/Avatar/network/members
[stars-shield]: https://img.shields.io/github/stars/3C-SCSU/Avatar.svg?style=for-the-badge
[stars-url]: https://github.com/3C-SCSU/Avatar/stargazers
[issues-shield]: https://img.shields.io/github/issues/3C-SCSU/Avatar.svg?style=for-the-badge
[issues-url]: https://github.com/3C-SCSU/Avatar/issues
[license-shield]: https://img.shields.io/github/license/3C-SCSU/Avatar.svg?style=for-the-badge
[license-url]: https://github.com/3C-SCSU/Avatar/blob/master/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/linkedin_username
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 

