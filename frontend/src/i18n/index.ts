import en from './en.json';
import fr from './fr.json';

export type Language = 'en' | 'fr';

const translations = { en, fr };

export function getTranslation(lang: Language) {
  return translations[lang] || translations.en;
}
