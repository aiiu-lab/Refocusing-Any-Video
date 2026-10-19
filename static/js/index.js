document.addEventListener('DOMContentLoaded', function () {
  var burgers = Array.prototype.slice.call(document.querySelectorAll('.navbar-burger'), 0);

  burgers.forEach(function (burger) {
    burger.addEventListener('click', function () {
      var menu = document.querySelector('.navbar-menu');
      burger.classList.toggle('is-active');
      if (menu) {
        menu.classList.toggle('is-active');
      }
    });
  });

  document.querySelectorAll('video').forEach(function (video) {
    video.controls = true;
  });

  document.querySelectorAll('video[autoplay]').forEach(function (video) {
    video.muted = true;
    video.playsInline = true;

    var playPromise = video.play();
    if (playPromise && typeof playPromise.catch === 'function') {
      playPromise.catch(function () {});
    }
  });

  document.querySelectorAll('[data-zoom-carousel]').forEach(function (carousel) {
    var slides = Array.prototype.slice.call(carousel.querySelectorAll('.zoom-carousel-slide'));
    var dots = Array.prototype.slice.call(carousel.querySelectorAll('.zoom-carousel-dots button'));
    var previousButton = carousel.querySelector('.zoom-carousel-arrow.is-left');
    var nextButton = carousel.querySelector('.zoom-carousel-arrow.is-right');
    var interval = parseInt(carousel.dataset.interval, 10) || 7000;
    var activeIndex = Math.max(slides.findIndex(function (slide) {
      return slide.classList.contains('is-active');
    }), 0);
    var timer = null;

    var playSlideVideos = function (slide) {
      slide.querySelectorAll('video').forEach(function (video) {
        video.muted = true;
        video.playsInline = true;
        try {
          video.currentTime = 0;
        } catch (error) {}
        var playPromise = video.play();
        if (playPromise && typeof playPromise.catch === 'function') {
          playPromise.catch(function () {});
        }
      });
    };

    var pauseSlideVideos = function (slide) {
      slide.querySelectorAll('video').forEach(function (video) {
        video.pause();
        try {
          video.currentTime = 0;
        } catch (error) {}
      });
    };

    var activateSlide = function (index) {
      activeIndex = (index + slides.length) % slides.length;

      slides.forEach(function (slide, slideIndex) {
        var isActive = slideIndex === activeIndex;
        slide.classList.toggle('is-active', isActive);
        slide.setAttribute('aria-hidden', isActive ? 'false' : 'true');
        if (isActive) {
          playSlideVideos(slide);
        } else {
          pauseSlideVideos(slide);
        }
      });

      dots.forEach(function (dot, dotIndex) {
        var isActive = dotIndex === activeIndex;
        dot.classList.toggle('is-active', isActive);
        if (isActive) {
          dot.setAttribute('aria-current', 'true');
        } else {
          dot.removeAttribute('aria-current');
        }
      });
    };

    var scheduleNextSlide = function () {
      window.clearTimeout(timer);
      timer = window.setTimeout(function () {
        activateSlide(activeIndex + 1);
        scheduleNextSlide();
      }, interval);
    };

    if (!slides.length) {
      return;
    }

    if (previousButton) {
      previousButton.addEventListener('click', function () {
        activateSlide(activeIndex - 1);
        scheduleNextSlide();
      });
    }

    if (nextButton) {
      nextButton.addEventListener('click', function () {
        activateSlide(activeIndex + 1);
        scheduleNextSlide();
      });
    }

    dots.forEach(function (dot, dotIndex) {
      dot.addEventListener('click', function () {
        activateSlide(dotIndex);
        scheduleNextSlide();
      });
    });

    activateSlide(activeIndex);
    if (slides.length > 1) {
      scheduleNextSlide();
    }
  });
});
