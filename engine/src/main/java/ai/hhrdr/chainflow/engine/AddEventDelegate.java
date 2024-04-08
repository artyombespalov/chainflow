package ai.hhrdr.chainflow.engine;

import net.minidev.json.JSONObject;
import net.minidev.json.parser.JSONParser;
import org.camunda.bpm.engine.delegate.DelegateExecution;
import org.camunda.bpm.engine.delegate.JavaDelegate;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpRequest.BodyPublishers;
import java.net.http.HttpResponse;
import java.net.http.HttpResponse.BodyHandlers;
import java.util.logging.Logger;

@Component("AddEventDelegate")
public class AddEventDelegate implements JavaDelegate {

    @Value("${api.url}")
    private String apiURL;

    @Value("${api.key}")
    private String apiKey;

    private static final Logger LOGGER = Logger.getLogger(AddEventDelegate.class.getName());

    @Override
    public void execute(DelegateExecution execution) throws Exception {
        String eventId = (String) execution.getVariable("event_id");
        String eventName = (String) execution.getVariable("event_name");
        String eventDescription = (String) execution.getVariable("event_description");
        String ingEventThumbnail = (String) execution.getVariable("img_event_thumbnail");
        HttpClient client = HttpClient.newHttpClient();
        JSONObject json = new JSONObject();
        json.put("event_id", eventId);
        json.put("event_name", eventName);
        json.put("event_description", eventDescription);
        json.put("img_event_thumbnail", ingEventThumbnail);
        HttpRequest request;// Ensure your API key is correctly set up for authorization
        if (eventId != null) {
            request = HttpRequest.newBuilder()
                    .uri(URI.create(apiURL + "/api/events/" + eventId))
                    .header("Content-Type", "application/json")
                    .header("Authorization", apiKey) // Ensure your API key is correctly set up for authorization
                    .PUT(BodyPublishers.ofString(json.toString()))
                    .build();
        } else {
            request = HttpRequest.newBuilder()
                    .uri(URI.create(apiURL + "/api/events"))
                    .header("Content-Type", "application/json")
                    .header("Authorization", apiKey) // Ensure your API key is correctly set up for authorization
                    .POST(BodyPublishers.ofString(json.toString()))
                    .build();
        }
        try {
            HttpResponse<String> response = client.send(request, BodyHandlers.ofString());

            String eventJson = response.body();
            JSONParser parser = new JSONParser(JSONParser.MODE_PERMISSIVE);
            Object obj = parser.parse(eventJson); // Use Object to hold the parsed result

            if (obj instanceof JSONObject) { // Check if the parsed object is indeed a JSONObject
                JSONObject event = (JSONObject) obj; // Safe casting to JSONObject

                String eventIdRes = (String) event.get("id"); // Now safely extract the 'id'
                execution.setVariable("eventId", eventIdRes);
                LOGGER.info("event creation response status code: " + response.statusCode());
                LOGGER.info("event creation response body: " + response.body());
            } else {
                LOGGER.severe("Parsed response is not a JSON object as expected.");
                // Handle this scenario appropriately (e.g., throw an exception or log a detailed message)
            }
        } catch (Exception e) {
            LOGGER.severe("Failed to add event. Exception: " + e.getMessage());
            throw e; // Re-throw if you want to propagate the error
        }

    }
}
