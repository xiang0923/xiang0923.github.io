
// enhancements.js - scroll progress, back-to-top, ripple, tooltips

document.addEventListener('DOMContentLoaded', () => {
  // Scroll Progress Bar
  const progressBar = document.createElement('div');
  progressBar.id = 'progress-bar';
  document.body.prepend(progressBar);
  window.addEventListener('scroll', () => {
    const scrollTop = window.pageYOffset;
    const docHeight = document.documentElement.scrollHeight - window.innerHeight;
    const scrollPercent = (scrollTop / docHeight) * 100;
    progressBar.style.width = scrollPercent + '%';
    // Back to top
    if (scrollTop > window.innerHeight) {
      backBtn.classList.add('show');
    } else {
      backBtn.classList.remove('show');
    }
  });

  // Back to top button
  const backBtn = document.createElement('button');
  backBtn.id = 'back-to-top';
  backBtn.innerHTML = '↑';
  document.body.appendChild(backBtn);
  backBtn.addEventListener('click', () => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });

  // Ripple effect on button clicks
  document.querySelectorAll('button').forEach(btn => {
    btn.classList.add('ripple-effect');
    btn.addEventListener('click', function(e) {
      const rect = this.getBoundingClientRect();
      const circle = document.createElement('span');
      const diameter = Math.max(rect.width, rect.height);
      const radius = diameter / 2;
      circle.style.width = circle.style.height = diameter + 'px';
      circle.style.left = e.clientX - rect.left - radius + 'px';
      circle.style.top = e.clientY - rect.top - radius + 'px';
      circle.style.position = 'absolute';
      circle.style.borderRadius = '50%';
      circle.style.background = 'rgba(255,255,255,0.3)';
      circle.style.transform = 'scale(0)';
      circle.style.opacity = '0.75';
      circle.style.pointerEvents = 'none';
      circle.style.transition = 'transform 0.5s, opacity 1s';
      this.appendChild(circle);
      setTimeout(() => {
        circle.style.transform = 'scale(1)';
        circle.style.opacity = '0';
      }, 0);
      setTimeout(() => circle.remove(), 1000);
    });
  });

  // Initialize tooltips
  document.querySelectorAll('[data-tooltip]').forEach(elem => {
    const wrapper = document.createElement('span');
    wrapper.className = 'tooltip';
    const tooltipText = document.createElement('span');
    tooltipText.className = 'tooltip-text';
    tooltipText.textContent = elem.getAttribute('data-tooltip');
    elem.parentNode.insertBefore(wrapper, elem);
    wrapper.appendChild(elem);
    wrapper.appendChild(tooltipText);
  });
});
