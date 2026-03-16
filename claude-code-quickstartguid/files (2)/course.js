// Shared navigation state
const navLinks = [
  { href: 'index.html',   label: 'Course Overview',   num: '00' },
  { href: 'module1.html', label: 'What is Claude Code?', num: '01' },
  { href: 'module2.html', label: 'Installation & Setup', num: '02' },
  { href: 'module3.html', label: 'Core Commands & Sessions', num: '03' },
  { href: 'module4.html', label: 'CLAUDE.md & Project Memory', num: '04' },
  { href: 'module5.html', label: 'Slash Commands & Workflows', num: '05' },
  { href: 'module6.html', label: 'MCP Integration', num: '06' },
  { href: 'module7.html', label: 'Hooks, Subagents & Advanced', num: '07' },
  { href: 'module8.html', label: 'Skills', num: '08' },
  { href: 'module9.html', label: 'GitHub Integration', num: '09' },
  { href: 'labs.html',    label: 'Practice Labs', num: '🧪' },
];

function buildNav(currentPage) {
  const sidebar = document.getElementById('sidebar');
  if (!sidebar) return;

  const logo = `
    <a href="index.html" class="sidebar-logo">
      <div class="logo-icon">CC</div>
      <div class="logo-text">
        <div class="title">Claude Code</div>
        <div class="sub">Complete Course</div>
      </div>
    </a>`;

  const links = navLinks.map(n => {
    const active = n.href === currentPage ? ' active' : '';
    return `<a href="${n.href}" class="nav-link${active}">
      <span class="num">${n.num}</span>
      ${n.label}
    </a>`;
  }).join('');

  sidebar.innerHTML = logo + `
    <div class="nav-section">
      <div class="nav-section-label">Modules</div>
      ${links}
    </div>
    <div style="margin-top:auto;padding:20px;border-top:1px solid var(--border)">
      <div style="font-size:12px;color:var(--text3);font-family:'JetBrains Mono',monospace">
        v2025 · 8 modules · 12+ labs
      </div>
    </div>`;
}

// Mobile menu
function initMobileMenu() {
  const toggle = document.querySelector('.menu-toggle');
  const sidebar = document.getElementById('sidebar');
  if (toggle && sidebar) {
    toggle.addEventListener('click', () => sidebar.classList.toggle('open'));
    document.addEventListener('click', e => {
      if (!sidebar.contains(e.target) && !toggle.contains(e.target)) {
        sidebar.classList.remove('open');
      }
    });
  }
}

document.addEventListener('DOMContentLoaded', initMobileMenu);
