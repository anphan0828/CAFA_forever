import { useState } from 'react'
import { Section } from '../layout'
import './HeroSection.css'

export function HeroSection() {
  const [showLearnMore, setShowLearnMore] = useState(false)

  return (
    <Section variant="hero" id="hero">
      <div className="hero">
        <h2 className="hero__acronym">
          <span className="hero__acronym-letter">L</span>ongitudinal{' '}
          <span className="hero__acronym-letter">A</span>ssessment of{' '}
          <span className="hero__acronym-letter">F</span>unction{' '}
          <span className="hero__acronym-letter">A</span>nnotation
        </h2>
        <p className="hero__description">
          A persistent benchmarking system for protein function prediction methods.
          Unlike CAFA's triennial challenges, LAFA continuously evaluates methods
          as new ground truth annotations accumulate.
        </p>

        <div className="hero__badges">
          <a
            href="https://arxiv.org/pdf/2604.20782"
            target="_blank"
            rel="noopener noreferrer"
            className="hero__badge"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
              <polyline points="14 2 14 8 20 8" />
              <line x1="16" y1="13" x2="8" y2="13" />
              <line x1="16" y1="17" x2="8" y2="17" />
              <polyline points="10 9 9 9 8 9" />
            </svg>
            Read the Paper
          </a>
          <a
            href="https://github.com/anphan0828/LAFA_container_guide"
            target="_blank"
            rel="noopener noreferrer"
            className="hero__badge hero__badge--secondary"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="17 8 12 3 7 8" />
              <line x1="12" y1="3" x2="12" y2="15" />
            </svg>
            Submit Your Method
          </a>
        </div>

        <button
          className="hero__learn-more-toggle"
          onClick={() => setShowLearnMore(!showLearnMore)}
          aria-expanded={showLearnMore}
        >
          {showLearnMore ? 'Hide Details' : 'How It Works'}
          <svg
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            className={showLearnMore ? 'hero__chevron--up' : ''}
          >
            <polyline points="6 9 12 15 18 9" />
          </svg>
        </button>

        {showLearnMore && (
          <div className="hero__learn-more">
            <div className="hero__learn-more-section">
              <h4>Gene Ontology & Function Prediction</h4>
              <p>
                The <strong>Gene Ontology (GO)</strong> is a structured vocabulary describing
                protein functions across three aspects: <em>Biological Process</em> (BP) describes
                cellular activities, <em>Molecular Function</em> (MF) describes biochemical
                activities, and <em>Cellular Component</em> (CC) describes subcellular locations.
                GO terms are organized in a directed acyclic graph where child terms inherit
                properties from their ancestors—if a protein performs a specific function,
                it implicitly performs all parent functions. Function prediction methods
                exploit this hierarchical structure to propagate predictions along the graph,
                ensuring consistency and leveraging relationships between terms.
              </p>
            </div>
            <div className="hero__learn-more-section">
              <h4>Evaluation Windows</h4>
              <p>
                An evaluation window spans two time points: <strong>Start (t₀)</strong> when
                predictions are made, and <strong>End (t₁)</strong> when accumulated
                annotations become ground truth. Longer windows provide more accumulated
                annotations for robust evaluation.
              </p>
            </div>
            <div className="hero__learn-more-section">
              <h4>Knowledge Subsets</h4>
              <p>
                Proteins are stratified by their annotation status at prediction time (t₀)
                to evaluate methods under different prior knowledge conditions. This separation
                is crucial because methods that rely on sequence homology or existing annotations
                may perform differently when such information is unavailable.
              </p>
              <dl className="hero__definitions">
                <div>
                  <dt>NK (No Knowledge)</dt>
                  <dd>
                    Proteins with no prior experimental GO annotations at t₀. This is the
                    most challenging subset, testing a method's ability to predict function
                    purely from sequence or structure without any functional hints.
                  </dd>
                </div>
                <div>
                  <dt>LK (Limited Knowledge)</dt>
                  <dd>
                    Proteins with some non-experimental (computational) annotations at t₀.
                    Methods can leverage these weaker signals, though they may be incomplete
                    or less reliable than experimental evidence.
                  </dd>
                </div>
                <div>
                  <dt>PK (Prior Knowledge)</dt>
                  <dd>
                    Proteins with existing experimental annotations in at least one GO aspect.
                    This tests a method's ability to transfer knowledge across aspects or
                    refine existing annotations with new predictions.
                  </dd>
                </div>
              </dl>
            </div>
          </div>
        )}
      </div>
    </Section>
  )
}
