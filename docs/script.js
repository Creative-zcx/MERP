const menuButton = document.querySelector('.menu-button');
const nav = document.querySelector('nav');

menuButton?.addEventListener('click', () => {
  const open = document.body.classList.toggle('menu-open');
  menuButton.setAttribute('aria-expanded', String(open));
});

nav?.querySelectorAll('a').forEach((link) => {
  link.addEventListener('click', () => {
    document.body.classList.remove('menu-open');
    menuButton?.setAttribute('aria-expanded', 'false');
  });
});

const observer = new IntersectionObserver((entries) => {
  entries.forEach((entry) => {
    if (entry.isIntersecting) {
      entry.target.classList.add('is-visible');
      observer.unobserve(entry.target);
    }
  });
}, { threshold: 0.12, rootMargin: '0px 0px -40px' });

document.querySelectorAll('.reveal').forEach((element) => observer.observe(element));

const header = document.querySelector('.site-header');
window.addEventListener('scroll', () => {
  header?.classList.toggle('is-scrolled', window.scrollY > 24);
}, { passive: true });

const visual = document.querySelector('.hero-visual');
if (visual && !window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
  visual.addEventListener('pointermove', (event) => {
    const box = visual.getBoundingClientRect();
    const x = (event.clientX - box.left) / box.width - 0.5;
    const y = (event.clientY - box.top) / box.height - 0.5;
    visual.style.setProperty('--tilt-x', `${y * -5}deg`);
    visual.style.setProperty('--tilt-y', `${x * 5}deg`);
  });
  visual.addEventListener('pointerleave', () => {
    visual.style.setProperty('--tilt-x', '0deg');
    visual.style.setProperty('--tilt-y', '0deg');
  });
}
