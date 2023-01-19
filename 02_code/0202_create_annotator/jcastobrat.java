package com.ibm.au.research.nlp.consumer;

import java.io.File;
import java.io.IOException;
import java.util.LinkedList;

import org.apache.uima.UimaContext;
import org.apache.uima.analysis_component.JCasAnnotator_ImplBase;
import org.apache.uima.analysis_engine.AnalysisEngineDescription;
import org.apache.uima.analysis_engine.AnalysisEngineProcessException;
import org.apache.uima.fit.factory.AnalysisEngineFactory;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.resource.ResourceInitializationException;
import org.cleartk.ne.type.NamedEntityMention;
import org.cleartk.util.ViewUriUtil;

import com.ibm.au.research.nlp.types.Event;
import com.ibm.au.research.nlp.types.EventRole;
import com.ibm.au.research.nlp.types.Relation;

import au.com.nicta.csp.brateval.Annotations;
import au.com.nicta.csp.brateval.Document;
import au.com.nicta.csp.brateval.Entity;
import au.com.nicta.csp.brateval.Location;

public class JCas2Brat extends JCasAnnotator_ImplBase {
	public static final String PARAM_OUTPUT_FOLDER_NAME = "outputFolderName";

	private String output_folder_name = "";

	public void initialize(UimaContext context) throws ResourceInitializationException {
		output_folder_name = (String) context.getConfigParameterValue(PARAM_OUTPUT_FOLDER_NAME);
	}

	@Override
	public void process(JCas jCas) throws AnalysisEngineProcessException {
		int entity_id = 1;
		int relation_id = 1;
		int event_id = 1;

		// Text file
		File textFile = new File(ViewUriUtil.getURI(jCas));

		// Annotation file
		String prefix = textFile.getName().replaceAll("[.]txt$", "");
		String annFile_name = prefix + ".ann";

		Document d = new Document();

		// Entities
		for (NamedEntityMention namedEntity : JCasUtil.select(jCas, NamedEntityMention.class)) {
			String entityId = "T" + entity_id;
			namedEntity.setMentionId(entityId);
			LinkedList<Location> location = new LinkedList<Location>();
			location.add(new Location(namedEntity.getBegin(), namedEntity.getEnd()));

			d.addEntity(entityId,
					new Entity(entityId, namedEntity.getMentionType(), location, namedEntity.getCoveredText(), "ann"));

			entity_id++;
		}

		// Relations
		for (Relation relation : JCasUtil.select(jCas, Relation.class)) {
			String relationId = "R" + relation_id;
			
			d.addRelation(relationId, new au.com.nicta.csp.brateval.Relation(relationId, relation.getRelationType(),
					"arg1", d.getEntity(relation.getArg1().getMentionId()), "arg2", d.getEntity(relation.getArg2().getMentionId()), "ann"));

			relation_id++;

		}

		/*
		 * // Events for (Event event : JCasUtil.select(jCas, Event.class)) {
		 * String eventId = "E" + event_id;
		 * 
		 * LinkedList <String> arguments = new LinkedList <String> ();
		 * 
		 * for (int i = 0; i < event.getRoles().size(); i++) { EventRole
		 * eventRole = event.getRoles(i); arguments.add(eventRole.getRole() +
		 * ":" + eventRole.getEntity().getMentionId()); }
		 * 
		 * d.addEvent(eventId, new au.com.nicta.csp.brateval.Event(eventId,
		 * event.getEventType(), event.getTrigger().getMentionId(), arguments,
		 * "ann"));
		 * 
		 * event_id++; }
		 */

		try {
			Annotations.write(new File(output_folder_name, annFile_name).getAbsolutePath(), d);
		} catch (IOException e) {
			new RuntimeException(e);
		}
	}

	public static AnalysisEngineDescription getOutputFolderDescription(String outputFolderName)
			throws ResourceInitializationException {
		// TODO Allow setting up the name of the Excel file to write
		return AnalysisEngineFactory.createEngineDescription(JCas2Brat.class, PARAM_OUTPUT_FOLDER_NAME,
				outputFolderName);
	}
}