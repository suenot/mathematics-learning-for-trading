import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

const isGitHubPages = !!process.env.GITHUB_PAGES;

const CHAPTERS = [
  {
    label: 'Chapter 1 — Stochastic Calculus',
    labelRu: 'Глава 1 — Стохастическое исчисление',
    dir: '01-stochastic-calculus',
  },
  {
    label: 'Chapter 2 — Market Microstructure',
    labelRu: 'Глава 2 — Микроструктура рынка',
    dir: '02-market-microstructure',
  },
  {
    label: 'Chapter 3 — Portfolio Optimization',
    labelRu: 'Глава 3 — Портфельная оптимизация',
    dir: '03-portfolio-optimization',
  },
  {
    label: 'Chapter 4 — ML for Time Series',
    labelRu: 'Глава 4 — ML для временных рядов',
    dir: '04-ml-time-series',
  },
  {
    label: 'Chapter 5 — Low Latency Systems',
    labelRu: 'Глава 5 — Системы низкой задержки',
    dir: '05-low-latency-systems',
  },
];

const sidebar = [
  {
    label: 'Overview',
    translations: { ru: 'Введение' },
    items: [
      { label: 'About the Book', link: '/', translations: { ru: 'О книге' } },
    ],
  },
  ...CHAPTERS.map(chapter => ({
    label: chapter.label,
    translations: { ru: chapter.labelRu },
    autogenerate: { directory: `chapters/${chapter.dir}` },
    collapsed: true,
  })),
];

export default defineConfig({
  site: isGitHubPages ? 'https://suenot.github.io' : 'https://suenot.com',
  ...(isGitHubPages && { base: '/mathematics-learning-for-trading-website' }),
  integrations: [
    starlight({
      title: 'Mathematics for Trading',
      description: 'Practical Guide to Mathematics in Algorithmic Trading',
      logo: {
        light: './src/assets/logo-light.svg',
        dark: './src/assets/logo-dark.svg',
        replacesTitle: false,
      },
      social: {
        github: 'https://github.com/suenot/mathematics-learning-for-trading',
      },
      defaultLocale: 'root',
      locales: {
        root: {
          label: 'English',
          lang: 'en',
        },
        ru: {
          label: 'Русский',
          lang: 'ru',
        },
      },
      sidebar,
      customCss: ['./src/styles/custom.css'],
    }),
  ],
});
